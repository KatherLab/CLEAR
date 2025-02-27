from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import BinaryAUROC
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt       
import re
import os

        
def get_class_weights(targets):
    pos_weights = []
    for i in range(targets.shape[1]):
        weight = len(targets[targets[:,i]==1])/len(targets[targets[:,i]==0])
        pos_weights.append(weight)
    return pos_weights

def train(model, train_loader, valid_loader, optimizer, model_name, odir, config):
    best_metric = float('inf')  # Initialize with a high value
    best_epoch = -1
    best_state_dict = None
    positive_weights = get_class_weights(train_loader.dataset.get_targets())
    positive_weights = torch.tensor(positive_weights, dtype=torch.float).cuda()
    
    loss_func = nn.BCEWithLogitsLoss(weight=positive_weights)
    for epoch in range(config.num_epochs):
        model.train()  # Set the model to training mode
        total_loss= 0.0
        total_train_correct=0.0
        total_val_correct=0.0
        train_preds=[]
        train_target=[]
        
        # Training Step
        # all_image_logits, all_text_logits = [], []
        for i, (image_features, idxs, labels, pt_id) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            #labels = labels-1
            image_features, idxs, labels = image_features.to(config.device), idxs.to(config.device), labels.to(config.device)#, attention_mask.to(config.device)                                 
            optimizer.zero_grad()  # Zero the gradients
            # Forward pass
            if config.model_name == 'ABMIL_Classification':
                logits, _ = model(image_features)#, idxs) #ABMIL
            else:
                logits = model(image_features)#, idxs)
            logits = logits.to(config.device)
            loss = loss_func(logits, labels).to(torch.float32)
            
            loss.backward()
            optimizer.step()  # Update weights
            total_loss += loss.item()           
            predicted_labels = (F.sigmoid(logits) > 0.5).to(torch.float32)
            total_train_correct += (predicted_labels == labels).sum().item()
            train_preds.extend(predicted_labels.cpu().numpy())
            train_target.extend(labels.cpu().numpy())
            
        train_preds = np.array(train_preds)
        train_target = np.array(train_target)
        train_acc = total_train_correct / len(train_loader)
        avg_loss = total_loss/len(train_loader)
        print(f'epoch {epoch}')    
        print(f'train accuracy: {train_acc:.4f}') 
        print(f'train loss: {avg_loss:.4f}')
        for val in range(np.array(train_preds).shape[-1]):
            metric = BinaryAUROC(thresholds=None)
            preds=torch.FloatTensor(train_preds[:,val]).to(config.device)
            target=torch.FloatTensor(train_target[:,val]).to(config.device)
            train_auc =metric(preds, target)
            print(f'train AUROC {config.csv_caption_key[val]}: {train_auc:.4f}')
        total_val_loss = 0.0
        #val_image_logits, val_text_logits = [], []
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_target = []
            for i, (image_features, idxs, labels, pt_id) in tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False):
                #labels = labels-1
                image_features, idxs, labels = image_features.to(config.device), idxs.to(config.device), labels.to(config.device)  
                # Forward pass
                if config.model_name == 'ABMIL_Classification':
                    logits, _ = model(image_features)#, idxs) #ABMIL
                else:
                    logits = model(image_features)#, idxs)
                val_loss =loss_func(logits, labels)#.to(torch.float32)

                total_val_loss += val_loss.item()
                
                predicted_labels = (F.sigmoid(logits) > 0.5).to(torch.float32)
                total_val_correct += (predicted_labels == labels).sum().item()
                
                val_preds.extend(predicted_labels.cpu().numpy())
                val_target.extend(labels.cpu().numpy())
        
        val_preds = np.array(val_preds)
        val_target = np.array(val_target)
                
        val_acc = total_val_correct / len(valid_loader)
        avg_valid_loss = total_val_loss/len(valid_loader)
        
        print(f'validation accuracy: {val_acc:.4f}')  
        print(f'val loss: {total_val_loss / len(valid_loader):.4f}')
        for val in range(np.array(val_preds).shape[-1]):
            metric = BinaryAUROC(thresholds=None)
            preds=torch.FloatTensor(val_preds[:,val]).to(config.device)
            target=torch.FloatTensor(val_target[:,val]).to(config.device)
            val_auc =metric(preds, target)
            print(f'val AUROC {config.csv_caption_key[val]}: {val_auc:.4f}')
        
        if avg_valid_loss < best_metric:
            best_metric = avg_valid_loss
            best_epoch = epoch+1
            best_state_dict = model.state_dict()
            #torch.save(model.state_dict(), f'{model_name}.pth')
            print(f"==== New best model found in {epoch+1} ===")

        if abs(epoch - best_epoch) == config.early_stopping:
            print(f"Early stopping triggered, no improvement since epoch {best_epoch}...")
            break
    torch.save(best_state_dict, f'{model_name}')

              
def evaluate(model, test_loader, model_name, model_path, odir, config):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_test_correct = 0.0
        pts = []
        y_true_class = []
        y_pred_class = []  
        log_scores = []    
        key_attscores=[]     
        
        for i, (image_features, idxs, labels, pt_id) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
           # labels = labels-1
            image_features, idxs, labels = image_features.to(config.device), idxs.to(config.device), labels.to(config.device)
            # Forward pass
            pts.extend(pt_id)
            if config.model_name == 'ABMIL_Classification':
                logits, att_scores  = model(image_features)#, idxs)
                key_att = [torch.argsort(att_scores.flatten(), descending=True)[:5].cpu().numpy()]
                key_attscores.extend(key_att)
            else:
                logits = model(image_features)#, idxs)

            y_true_class.extend(labels.cpu().numpy())
            
            predicted_labels = torch.argmax(logits, dim=1)
            predicted_labels = (F.sigmoid(logits) > 0.5).to(torch.float32)
            log_scores.append(logits.cpu().numpy().tolist())
            total_test_correct += (predicted_labels == labels).sum().item()
            y_pred_class.extend(predicted_labels.cpu().numpy())
            
        preds_df = pd.DataFrame({'pt_id': pts})
            
        y_pred_class = np.array(y_pred_class)
        y_true_class = np.array(y_true_class)
        log_scores = np.array(log_scores).squeeze(1)
        
        test_acc = total_test_correct / len(test_loader)
        # output
        print(f'Test accuracy: {test_acc:.4f}')  
        
        
        for idx,val in enumerate(range(np.array(y_pred_class).shape[-1])):
            metric = BinaryAUROC(thresholds=None)
            preds=torch.FloatTensor(y_pred_class[:,val]).to(config.device)
            target=torch.FloatTensor(y_true_class[:,val]).to(config.device)
            val_auc = metric(preds, target)
            print(f'test AUROC {config.csv_caption_key[val]}: {val_auc:.4f}')
            ax = plt.subplot()
            metrics.ConfusionMatrixDisplay.from_predictions(target.cpu().numpy(),preds.cpu().numpy(),normalize='true')
            ax.set_ylabel('Predicted labels');ax.set_xlabel('True labels'); 
            ax.set_title(f'Confusion Matrix {config.csv_caption_key[val]}'); 
            plt.savefig(f'{odir}/conf_matrix_{config.csv_caption_key[val]}.png')  
            preds_df[f'true_label_{config.csv_caption_key[val]}'] =  target.cpu().numpy()  
            preds_df[f'pred_{config.csv_caption_key[val]}'] = preds.cpu().numpy()  
            preds_df[f'logscore_{config.csv_caption_key[val]}'] = log_scores[:,idx]
        #preds_df[f'logscores'] =  log_scores
                                     
        #print(f'Test loss: {total_test_loss / len(valid_loader):.4f}')      
        preds_df['pt_id'] = preds_df['pt_id'].str.replace('/','_').str.strip('.png') 
        preds_df.to_csv(f'{odir}/patient_preds.csv')
        
        
        

def evaluateDL(model, test_loader, model_name, model_path, odir, config):
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    with torch.no_grad():
        total_test_correct = 0.0
        pts = []
        y_true_class = []
        y_pred_class = []  
        log_scores = []    
        key_attscores=[]     
        key_slices_att = {}
        keyslicedf = pd.read_csv('/path/to/DL_info_KS.csv')
        keyslicedf['pt_id']= keyslicedf['File_name'].str.extract('(\w+)_(\w+)')[0]
        for i, (image_features, idxs, labels, pt_id) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
           # labels = labels-1
            image_features, idxs, labels = image_features.to(config.device), idxs.to(config.device), labels.to(config.device)
            # Forward passç
            pts.extend(pt_id)
            if config.model_name == 'ABMIL_Classification':
                logits, att_scores = model(image_features) #, idxs)
                key_att = [torch.argsort(att_scores.flatten(), descending=True)[:5].cpu().numpy()]
                key_attscores.extend(key_att)
            else:
                logits = model(image_features)#, idxs)
            
            pt_info = keyslicedf[keyslicedf['pt_id']==pt_id[0]]
            
            key_slices = list(pt_info['New_index'].values)
            
            y_true_class.extend(labels.cpu().numpy())
            
            predicted_labels = torch.argmax(logits, dim=1)
            predicted_labels = (F.sigmoid(logits) > 0.5).to(torch.float32)
            log_scores.append(logits.cpu().numpy().tolist())
            total_test_correct += (predicted_labels == labels).sum().item()
            y_pred_class.extend(predicted_labels.cpu().numpy())
            if config.model_name == 'ABMIL_Classification': #plot attention
                os.makedirs(f'{odir}/attentionplots_v3', exist_ok=True)
                if len(idxs[idxs != -1].cpu().numpy()) > 50:
                    plt.figure(figsize=(16,9))
                    # Get attention scores and indices
                    att_soft = torch.softmax(att_scores[idxs != -1], dim=0).cpu().numpy().squeeze(1)
                    indices = idxs[idxs != -1].cpu().numpy()
                    if len(indices) == 512:
                        continue
                    # Separate even and odd positions
                    even_indices = indices[::2]
                    odd_indices = indices[1::2]
                    even_scores = att_soft[::2]
                    odd_scores = att_soft[1::2]
                    
                    fig, ax = plt.subplots(facecolor='#f0f0f0')
                    ax.set_facecolor('#f0f0f0')
                    # Plot smooth lines
                    # Plot lines with shaded areas
                   
                    # plt.fill_between(even_indices, even_scores, 0, alpha=0.3, color='#3498db', label='_nolegend_')
                    plt.plot(even_indices, even_scores, '-', color='#3498db', label='Clip abd', alpha=0.8, linewidth=2)

                    # plt.fill_between(odd_indices, odd_scores, 0, alpha=0.3, color='#e74c3c', label='_nolegend_')
                    plt.plot(odd_indices, odd_scores, '-', color='#e74c3c', label='Clip chest', alpha=0.8, linewidth=2)

                    # Add vertical lines for key slices
                    if key_slices:
                        for idx, val in enumerate(key_slices):
                            even_att = even_scores[val] 
                            odd_att = odd_scores[val]
                            key_slices_att[val] = max(even_att, odd_att)
                            plt.vlines(val*2, min(att_soft), max(att_soft), colors='#1a237e', 
                                    linestyles='--', alpha=0.5)

                    # Customize plot appearance
                    plt.ylabel('Attention scores', fontsize=10)
                    plt.xlabel('slices', fontsize=10)
                    plt.title(f'Attention distribution {pt_id}', fontsize=12, pad=15)

                    # Customize grid
                    plt.grid(True, alpha=0.3, linestyle='--')

                    # Customize legend
                    plt.legend(framealpha=0.8, facecolor='white', edgecolor='none')

                    # Adjust layout and save
                    plt.tight_layout()
                    #plt.savefig(f'{odir}/attentionplots_v2/attdistribution_{pt_id[0]}_{key_slices[idx]}.png')
                    plt.savefig(f'{odir}/attentionplots_v3/attdistribution_{pt_id[0]}_{key_slices[idx]}.png', dpi=300, bbox_inches='tight')
        

        if key_attscores:
            preds_df = pd.DataFrame({'pt_id': pts, 'attscores': key_attscores})
        else:
            preds_df = pd.DataFrame({'pt_id': pts})
                
        y_pred_class = np.array(y_pred_class)
        y_true_class = np.array(y_true_class)
        log_scores = np.array(log_scores).squeeze(1)
        
        test_acc = total_test_correct / len(test_loader)
        # output
        print(f'Test accuracy: {test_acc:.4f}')  
        
        
        for idx,val in enumerate(range(np.array(y_pred_class).shape[-1])):
            metric = BinaryAUROC(thresholds=None)
            preds=torch.FloatTensor(y_pred_class[:,val]).to(config.device)
            target=torch.FloatTensor(y_true_class[:,val]).to(config.device)
            val_auc = metric(preds, target)
            print(f'test AUROC {config.csv_caption_key[val]}: {val_auc:.4f}')
            ax = plt.subplot()
            metrics.ConfusionMatrixDisplay.from_predictions(preds.cpu().numpy(), target.cpu().numpy(),normalize='true')
            ax.set_ylabel('Predicted labels');ax.set_xlabel('True labels'); 
            ax.set_title(f'Confusion Matrix {config.csv_caption_key[val]}'); 
            plt.savefig(f'{odir}/conf_matrix_{config.csv_caption_key[val]}.png')  
            preds_df[f'true_label_{config.csv_caption_key[val]}'] =  target.cpu().numpy()  
            preds_df[f'pred_{config.csv_caption_key[val]}'] = preds.cpu().numpy()  
            preds_df[f'logscore_{config.csv_caption_key[val]}'] = log_scores[:,idx]
        #preds_df[f'logscores'] =  log_scores
        #preds_df['pt_id'] = preds_df['patient'].str.replace('/','_').str.strip('.png') 
        key_slices_df = pd.DataFrame.from_dict(key_slices_att, orient='index', columns=['attention'])
        key_slices_df.to_csv(f'{odir}/attention_keyslices.csv')
        preds_df.to_csv(f'{odir}/patient_preds_deploy.csv')
        
        
        
        
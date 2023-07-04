import sys
import torch
import os
import argparse
from glob import glob as gg
from sklearn.metrics import precision_score, recall_score, roc_auc_score, auc, precision_recall_curve, roc_curve
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
current_path = os.getcwd()
sys.path.append(current_path)
from AD_pretrained_utilities import CNN_8CL_B, CNN, torch_norm, loader, predict, \
    plot_complete_report, plot_auc_curve, plot_confusion_matrix


def test(net, batch_size, data_path, saver_path, img_preprocessing=False, device=torch.device('cpu')):
    if not os.path.exists(saver_path):
        os.makedirs(saver_path)
    # Load data
    dataset = gg(data_path + '/*.npy')
    if len(dataset) == 0:
        print('Empty folder. Please choose another folder containin npy files to process.')
        sys.exit()
    test_data = loader(dataset, transform=torch_norm, preprocessing=img_preprocessing)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    # Make predictions
    y_test, yp0, yp1, y_pred = predict(net, test_loader, device)
    # Evaluate data
    precision_0 = precision_score(y_test, y_pred, pos_label=0)
    precision_1 = precision_score(y_test, y_pred, pos_label=1)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    f1_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    roc_auc_0 = roc_auc_score(y_test, yp0)
    roc_auc_1 = roc_auc_score(y_test, yp1)
    prec_0, rec_0, _ = precision_recall_curve(y_test, yp0, pos_label=0)
    prec_1, rec_1, _ = precision_recall_curve(y_test, yp1)
    auprc_0 = auc(rec_0, prec_0)
    auprc_1 = auc(rec_1, prec_1)
    fpr, tpr, _ = roc_curve(y_test, yp1)

    # Save data and results
    df_report = pd.DataFrame({'Precision': [precision_0, precision_1],
                              'Recall': [recall_0, recall_1], 'F1-score': [f1_0, f1_1],
                              'AUC': [roc_auc_0, roc_auc_1], 'AUPRC': [auprc_0, auprc_1]})
    df_report.to_csv(saver_path + '/complete_report.csv')
    df_results = pd.DataFrame({'Accuracy': [acc], 'F1-score': [f1], 'AUC': [roc_auc_1]})
    df_results.to_csv(saver_path + '/Results.csv')
    logfile = open(saver_path + '/results.txt', 'w')
    logfile.write('Evaluation of (8CL, B) model pretrained on AD.\n\n')
    logfile.write('Dataset:' + data_path + '\n\n')
    logfile.write('F1-score: {:.4f}\n'.format(f1))
    logfile.write('Accuracy: {:.4f}\n'.format(acc))
    logfile.write('AUC: {:.4f}\n.'.format(roc_auc_1))
    logfile.flush()
    logfile.close()

    # PLOTS
    plot_complete_report(df_report, saver_path)
    plot_auc_curve(fpr, tpr, roc_auc_1, saver_path)
    plot_confusion_matrix(y_test, y_pred, saver_path)

# ######################
#   CONFIGURATION     ##
# ######################


source = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config = CNN_8CL_B()
model = CNN(config).to(source)
w = torch.load(current_path + '/AD_pretrained_weights.pt')
model.load_state_dict(w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Extract embeddings from the 3D-CNN (8CL, B) model pretrained on AD.
    File from <data_dir> are loaded and used as input for the model. The embedding <embedding> is extracted and saved in
    <saver_dir>.""")
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Data batch size. Default=1.')
    parser.add_argument('--data_dir', required=True, type=str,
                        help='The directory that contains npy files to be processed')
    parser.add_argument('--saver_dir', default=current_path + 'embeddings_AD_pretrained/', type=str,
                        help='The directory where to save the extracted embeddings')
    parser.add_argument('--preprocessing', default=False, type=bool,
                        help='If True, images are  directory where to save the extracted embeddings')
    args = parser.parse_args()
    test(model, args.batch_size, args.data_dir, args.saver_dir, args.preprocessing, source)

import numpy as np
from glob import glob as gg
import sys
import torch
import os
import argparse
from alive_progress import alive_bar
current_path = os.getcwd()
sys.path.append(current_path)
from AD_pretrained_utilities import CNN_8CL_B, CNN, torch_norm, img_processing
tonpy = lambda x: x.detach().cpu().numpy()


def extract_embedding(data_dir, embedding, saver_dir, net, device=torch.device('cpu'), processing=False):
    if not os.path.exists(saver_dir + '/L' + str(embedding+1) + '/'):
        os.makedirs(saver_dir + '/L' + str(embedding+1) + '/')
    files = gg(data_dir + '/*.npy')
    print('\nNumber of files to process:', len(files))
    n_emb = len(net.embedding)
    with alive_bar(len(files), bar='classic', spinner='arrow') as bar:
        for f in files:
            name = os.path.basename(f)
            x = np.load(f, allow_pickle=True)
            if processing:
                x = img_processing(x)
            x = torch_norm(x)[None, :]
            x = x.to(device)
            for i in range(n_emb):
                x = net.embedding[i](x)
                if i == embedding:
                    representation = tonpy(x.view(x.size(0), -1))[0]
                    np.save(saver_dir + '/L' + str(embedding+1) + '/' + name, representation)
                    bar()
                    break


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
    parser.add_argument('--data_dir', required=True, type=str,
                        help='The directory that contains npy files to be processed')
    parser.add_argument('--embedding', required=True, type=int,
                        help='The embedding to extract (counting from 0)')
    parser.add_argument('--saver_dir', default=current_path + 'embeddings_AD_pretrained/', type=str,
                        help='The directory where to save the extracted embeddings')
    parser.add_argument('--preprocessing', default=False, type=bool,
                        help='If True, images are  directory where to save the extracted embeddings')
    args = parser.parse_args()
    extract_embedding(args.data_dir, args.embedding, args.saver_dir, model, source, args.preprocessing)

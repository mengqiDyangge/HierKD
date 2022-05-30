from mmcv.fileio.handlers import base
from torch import random
import torch.nn as nn
import clip
import json
import torch

if __name__ == '__main__':
    # initial the weight of word embedding
    base_class = 'coco/coco_seen_sort.json'
    target_class = 'coco/coco_unseen_sort.json'
    all_class = 'coco/coco_seen+unseen_sort.json'

    prompt = 'a photo of a {}.'

    with open(base_class, 'r') as f:
        base_class = json.load(f)
    
    with open(target_class, 'r') as f:
        target_class = json.load(f)
    
    with open(all_class, 'r') as f:
        all_class = json.load(f)

    base_class = [prompt.format(word) for word in base_class]
    target_class = [prompt.format(word) for word in target_class]
    all_class = [prompt.format(word) for word in all_class]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    with torch.no_grad():
        base_class = clip.tokenize(base_class).to(device)
        base_features = model.encode_text(base_class)
        base_features = base_features / (base_features.norm(dim=-1, keepdim=True) + 1e-10)
        base_features = base_features.float()

        target_class = clip.tokenize(target_class).to(device)
        target_features = model.encode_text(target_class)
        target_features = target_features / (target_features.norm(dim=-1, keepdim=True) + 1e-10)
        target_features = target_features.float()

        all_class = clip.tokenize(all_class).to(device)
        all_features = model.encode_text(all_class)
        all_features = all_features / (all_features.norm(dim=-1, keepdim=True) + 1e-10)
        all_features = all_features.float()
    
    word_embedding_state_dict = {}
    word_embedding_state_dict['base'] = nn.Parameter(base_features)
    word_embedding_state_dict['target'] = nn.Parameter(target_features)
    word_embedding_state_dict['all'] = nn.Parameter(all_features)
    torch.save(word_embedding_state_dict, 'coco/word_embedding.pth')

    random_word_embedding_state_dict = {}
    random_base_feats = torch.rand(base_features.shape)
    random_base_feats = random_base_feats / (random_base_feats.norm(dim=-1, keepdim=True))
    random_target_feats = torch.rand(target_features.shape)
    random_target_feats = random_target_feats / (random_target_feats.norm(dim=-1, keepdim=True))
    random_all_feats = torch.rand(all_features.shape)
    random_all_feats = random_all_feats / (random_all_feats.norm(dim=-1, keepdim=True))

    random_word_embedding_state_dict['base'] = nn.Parameter(random_base_feats)
    random_word_embedding_state_dict['target'] = nn.Parameter(random_target_feats)
    random_word_embedding_state_dict['all'] = nn.Parameter(random_all_feats)
    torch.save(random_word_embedding_state_dict, 'coco/random_word_embedding.pth')

    import ipdb
    ipdb.set_trace()

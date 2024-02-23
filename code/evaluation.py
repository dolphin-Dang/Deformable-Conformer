from models import Conformer
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import numpy as np

config = {
    'res_path': './results/sub_result.txt',
    'sub_res_path': "./results/log_subject%d.txt",
    
    # test config
    'nSub': 1,
    'model_pth': './results_lyh/sub1/model.pth',
    'test_dir': './data/lyh_data/',

    # model config
    'emb_size': 40,
    'encoder_depth': 6,
    'decoder_depth': 3,
    'n_classes': 4,
    'channel': 14,
    
    'encoder_config': {
            'num_heads': 8,
            'drop_p': 0.5,
            'forward_expansion': 4,
            'forward_drop_p': 0.5,
            'num_of_points': 16
        },
    
    'decoder_config': {
            'num_heads': 8,
            'drop_p': 0.5,
            'forward_expansion': 4,
            'forward_drop_p': 0.5,
            'num_of_points': 12
        },
    
    'hidden_size_1': 256,
    'hidden_size_2': 64,
    'drop_p_1': 0.5,
    'drop_p_2': 0.3,
    
}

def get_test_data():
    left = np.load(config["test_dir"]+"left_t.npy")
    right = np.load(config["test_dir"]+"right_t.npy")
    leg = np.load(config["test_dir"]+"leg_t.npy")
    eeg = [left, right, leg]
    eeg = [t[:14, :] for t in eeg]
    X_test = []
    y_test = []

    for i in range(3):
        # print(f"Data shape: {eeg_raw[i].shape}")
        split_data = np.split(eeg[i], 500, axis=1)
        X_raw = np.stack(split_data, axis=0) # (14, 50_0000) => (500, 14, 1000)
        X_raw = np.expand_dims(X_raw, axis=1) # (500, 1, 14, 1000)
        y_raw = np.array([i for j in range(500)]) # (500,) value = label

        X_test.append(X_raw)
        y_test.append(y_raw)

    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    print(X_test.shape)
    print(y_test.shape)
    return X_test, y_test

def main():
    device = torch.device('cuda')
    model = Conformer(config=config).to(device)
    param_dict = torch.load(config["model_pth"], map_location=device)
    # print(param_dict.keys())
    model.load_state_dict(param_dict, strict=True)
    model.eval()
    
    X, y = get_test_data()

    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).long().to(device)
    
    # 输入数据到模型，得到预测结果
    outputs = model(X)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 计算损失
    loss_test = criterion(outputs, y)
    
    # 计算准确率
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
    correct = (predicted == y).sum().item()
    total = y.size(0)
    acc = correct / total
    
    cm = confusion_matrix(y.cpu(), predicted.cpu())
    print(cm)
    print('  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
          '  Test accuracy is %.6f' % acc)

if __name__ == "__main__":
    main()

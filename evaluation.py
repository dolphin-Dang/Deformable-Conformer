from conformer import ExP, Conformer
import torch
import torch.nn as nn

config = {
    'nSub': 1,
    'model_pth': './result_models/model_sub1.pth'
}


def main():
    model = Conformer()
    model.load_state_dict(torch.load(config["model_pth"]))
    model.eval()
    
    exp = ExP(config["nSub"])
    test_data, test_label = exp.get_test_data()
    test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)
    
    criterion_cls = torch.nn.CrossEntropyLoss().cuda()
    Tok, Cls = model(test_data)
    loss_test = criterion_cls(Cls, test_label)
    y_pred = torch.max(Cls, 1)[1]
    acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))

    print('  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
          '  Test accuracy is %.6f' % acc)

if __name__ == "__main__":
    main()

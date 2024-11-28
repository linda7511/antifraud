import copy


def load_lpa_subtensor(node_feat, work_node_feat, labels, seeds, input_nodes, device):
    # 准备图神经网络训练过程中的一批输入数据
    # 选择批次输入特征（数值特征）
    batch_inputs = node_feat[input_nodes].to(device)
    # 选择工作批次输入特征（类别特征）
    batch_work_inputs = {i: work_node_feat[i][input_nodes].to(
        device) for i in work_node_feat if i not in {"Labels"}}
    # for i in batch_work_inputs:
    #    print(batch_work_inputs[i].shape)
    # 选择批次标签
    batch_labels = labels[seeds].to(device)
    # 复制并修改标签
    train_labels = copy.deepcopy(labels)
    propagate_labels = train_labels[input_nodes]#从复制的标签中选择 input_nodes 指定的节点标签
    propagate_labels[:seeds.shape[0]] = 2#将前 seeds.shape[0] 个标签设置为 2

    # 批次输入特征，其他批次工作输入特征，批次标签，和修改后的标签
    return batch_inputs, batch_work_inputs, batch_labels, propagate_labels.to(device)

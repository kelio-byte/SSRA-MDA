import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, Linear, RGCNConv

warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


# ==================== 数据加载 ====================
class DataLoader:
    def __init__(self, dataset='HMDAD'):
        self.dataset = dataset
        self.adj_path = f"/{dataset}/mda.csv"
        self.mmi_path = f"/{dataset}/mm_sim.txt"
        self.ddi_path = f"/{dataset}/dd_sim.txt"
        self.m_embed_path = f"/{dataset}/microbe_embeddings.csv"
        self.d_embed_path = f"/{dataset}/disease_embeddings.csv"

    def load_data(self):
        """加载MDA矩阵和相似性矩阵"""
        # 加载MDA矩阵
        mda = pd.read_csv(self.adj_path, index_col=0, sep=",")
        mda = mda.values.astype(np.int64)
        microbe_idx, disease_idx = np.where(mda == 1)
        edges = np.stack([microbe_idx, disease_idx], axis=1)

        # 加载相似性矩阵
        mm_sim = np.loadtxt(self.mmi_path)
        dd_sim = np.loadtxt(self.ddi_path)
        print(f"Microbe similarity matrix shape: {mm_sim.shape}")
        print(f"Disease similarity matrix shape: {dd_sim.shape}")

        # 加载llm嵌入向量
        m_emb = pd.read_csv(self.m_embed_path, index_col=0).values
        d_emb = pd.read_csv(self.d_embed_path, index_col=0).values
        print(f"Microbe embedding matrix shape: {m_emb.shape}")
        print(f"Disease embedding matrix shape: {d_emb.shape}")

        return mda, edges, mm_sim, dd_sim, m_emb, d_emb


class GraphBuilder:
    def __init__(self, edges, mm_sim, dd_sim, m_emb, d_emb, d_k=15, m_k=25):
        self.edges = edges
        self.mm_sim = mm_sim
        self.dd_sim = dd_sim
        self.m_emb = m_emb
        self.d_emb = d_emb
        self.n_microbes = mm_sim.shape[0]
        self.n_diseases = dd_sim.shape[0]

        self.d_k = d_k
        self.m_k = m_k

    def build_graph_by_topk(self, sim_matrix, k):
        """使用topk构建图"""
        n_nodes = sim_matrix.shape[0]
        sim_matrix = sim_matrix.copy()
        np.fill_diagonal(sim_matrix, -1e9)

        k = min(k, n_nodes - 1) if n_nodes > 1 else 0
        if k <= 0:
            return np.zeros((0, 2), dtype=np.int64)

        idx = np.argpartition(-sim_matrix, kth=k, axis=1)[:, :k]
        rows = np.repeat(np.arange(n_nodes), k)
        cols = idx.reshape(-1)
        edges = np.stack([rows, cols], axis=1).astype(np.int64)
        edges = np.unique(edges, axis=0)
        edges = torch.LongTensor(edges)
        return edges

    def build_heterogeneous_graph(self):
        """构建异质图（M-D-M）"""
        hetero_data = HeteroData()
        hetero_data['microbe'].x_sim = torch.from_numpy(self.mm_sim.astype(np.float32))
        hetero_data['disease'].x_sim = torch.from_numpy(self.dd_sim.astype(np.float32))
        hetero_data['microbe'].x_sem = torch.from_numpy(self.m_emb.astype(np.float32))
        hetero_data['disease'].x_sem = torch.from_numpy(self.d_emb.astype(np.float32))

        # 加 MM，DD
        dd_edges = self.build_graph_by_topk(self.dd_sim, self.d_k)
        mm_edges = self.build_graph_by_topk(self.mm_sim, self.m_k)
        dd = torch.tensor(dd_edges.T, dtype=torch.long)
        mm = torch.tensor(mm_edges.T, dtype=torch.long)
        hetero_data['disease', 'similar', 'disease'].edge_index = dd
        hetero_data['microbe', 'similar', 'microbe'].edge_index = mm

        # 加 M-D
        m_d_edges = torch.tensor(self.edges.T, dtype=torch.long)
        hetero_data['microbe', 'associated', 'disease'].edge_index = torch.LongTensor(m_d_edges)

        # 反向边
        hetero_data['disease', 'rev_associated', 'microbe'].edge_index = torch.LongTensor(m_d_edges[[1, 0], :])

        return hetero_data


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE with in-batch negatives. Assumes one-to-one alignment."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    labels = torch.arange(z1.size(0), device=z1.device)

    logits12 = (z1 @ z2.t()) / tau
    loss12 = F.cross_entropy(logits12, labels)

    logits21 = (z2 @ z1.t()) / tau
    loss21 = F.cross_entropy(logits21, labels)

    return 0.5 * (loss12 + loss21)


# ==================== 主模型 ====================
class MDAPredictor(nn.Module):
    def __init__(self, in_m_sim, in_m_sem, in_d_sim, in_d_sem, hidden_dim=256, embed_dim=256, dropout=0.2,
                 num_relations=4):
        super(MDAPredictor, self).__init__()

        # 不管是同质图还是异质图，都先做线性投影，且权重共享
        self.m_sim_proj = nn.Sequential(
            Linear(in_m_sim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), Linear(hidden_dim, hidden_dim)
        )
        self.m_sem_proj = nn.Sequential(
            Linear(in_m_sem, hidden_dim), nn.ReLU(), nn.Dropout(dropout), Linear(hidden_dim, hidden_dim)
        )
        self.d_sim_proj = nn.Sequential(
            Linear(in_d_sim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), Linear(hidden_dim, hidden_dim)
        )
        self.d_sem_proj = nn.Sequential(
            Linear(in_d_sem, hidden_dim), nn.ReLU(), nn.Dropout(dropout), Linear(hidden_dim, hidden_dim)
        )

        # 同质图编码器
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, embed_dim)

        # 异质图编码器
        self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_dim, embed_dim, num_relations=num_relations)

        # 对比学习
        self.cl_loss_m = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), Linear(embed_dim, embed_dim))
        self.cl_loss_d = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), Linear(embed_dim, embed_dim))

        # 融合层
        self.m_fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU())
        self.d_fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU())

        # 双线性解码
        self.W = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.W)
        self.P = nn.Parameter(torch.empty(embed_dim, 64))
        self.Q = nn.Parameter(torch.empty(embed_dim, 64))
        nn.init.xavier_uniform_(self.P)
        nn.init.xavier_uniform_(self.Q)

        self.dropout = dropout

    def homo_encode(self, data, microbe_x, disease_x):
        mm_edge = data['microbe', 'similar', 'microbe'].edge_index
        x_microbe = self.gcn1(microbe_x, mm_edge)
        x_microbe = F.relu(x_microbe)
        x_microbe = F.dropout(x_microbe, p=self.dropout, training=self.training)
        x_microbe = self.gcn2(x_microbe, mm_edge)

        dd_edge = data['disease', 'similar', 'disease'].edge_index
        x_disease = self.gcn1(disease_x, dd_edge)
        x_disease = F.relu(x_disease)
        x_disease = F.dropout(x_disease, p=self.dropout, training=self.training)
        x_disease = self.gcn2(x_disease, dd_edge)

        return {"microbe": x_microbe, "disease": x_disease}

    def hetero_encode(self, data, microbe_x, disease_x):
        data = data.clone()

        data["microbe"].x = microbe_x
        data["disease"].x = disease_x

        # 转为 homogeneous 多关系图
        homo = data.to_homogeneous(node_attrs=["x"])  # x, edge_index, edge_type, node_type
        x = homo.x
        edge_index = homo.edge_index
        edge_type = homo.edge_type  # [num_edges]

        # 两层 R-GCN
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)

        # 按 node_type 切回 microbe / disease
        node_type = homo.node_type  # [num_nodes], 每个值是类型 id（顺序与 data.node_types 一致）
        microbe_type = data.node_types.index("microbe")
        disease_type = data.node_types.index("disease")

        z_microbe = x[node_type == microbe_type]
        z_disease = x[node_type == disease_type]

        return {"microbe": z_microbe, "disease": z_disease}

    def encode(self, data):
        """编码得到嵌入"""
        m_sim = self.m_sim_proj(data["microbe"].x_sim)
        d_sim = self.d_sim_proj(data["disease"].x_sim)
        z_sim = self.homo_encode(data, m_sim, d_sim)

        m_sem = self.m_sem_proj(data["microbe"].x_sem)
        d_sem = self.d_sem_proj(data["disease"].x_sem)
        z_sem = self.hetero_encode(data, m_sem, d_sem)
        # z_sem = self.homo_encode(data, m_sem, d_sem)

        return z_sim, z_sem

    def decode(self, z_sim, z_sem):
        """解码得到预测矩阵"""
        z_m = self.m_fusion(torch.cat([z_sim["microbe"], z_sem["microbe"]], dim=-1))
        z_d = self.d_fusion(torch.cat([z_sim["disease"], z_sem["disease"]], dim=-1))

        # logits = (z_m @ self.W) @ z_d.t()
        logits = (z_m @ self.P) @ (z_d @ self.Q).t()
        return logits

    def ori_cl_loss(self, z_sim, z_sem, tau):
        m1 = F.dropout(z_sim["microbe"], p=self.dropout, training=self.training)
        m2 = F.dropout(z_sem["microbe"], p=self.dropout, training=self.training)
        loss_m = info_nce(self.cl_loss_m(m1), self.cl_loss_m(m2), tau)

        d1 = F.dropout(z_sim["disease"], p=self.dropout, training=self.training)
        d2 = F.dropout(z_sem["disease"], p=self.dropout, training=self.training)
        loss_d = info_nce(self.cl_loss_d(d1), self.cl_loss_d(d2), tau)
        return loss_m + loss_d

    def neighbor_readout(self, z, edge_index, num_nodes):
        """
        z: [N, d]  node embeddings
        edge_index: [2, E]
        num_nodes: int
        """
        device = z.device
        edge_index = edge_index.to(device)

        row, col = edge_index

        # ---- 安全检查（CUDA 必需）----
        assert row.max().item() < num_nodes
        assert col.max().item() < num_nodes

        deg = torch.zeros((num_nodes,), device=device, dtype=z.dtype)
        deg.index_add_(0, row, torch.ones_like(row, dtype=z.dtype, device=device))
        # 归一化权重：1 / sqrt(deg_i * deg_j)
        weight = 1.0 / torch.sqrt(deg[row] * deg[col] + 1e-6)

        out = torch.zeros((num_nodes, z.size(1)), device=device, dtype=z.dtype)
        # out.index_add_(0, row, z[col])
        out.index_add_(0, row, z[col] * weight.unsqueeze(-1))

        # deg = deg.clamp(min=1.0).unsqueeze(-1)
        # return out / deg
        return out

    def cross_neighbor_readout(self, z_src, z_dst, edge_index, num_dst):
        """
        z_src: [N_src, d]  source node embeddings
        z_dst: [N_dst, d]  destination node embeddings
        edge_index: [2, E]  (src -> dst)
        num_dst: int
        """
        device = z_dst.device
        edge_index = edge_index.to(device)

        src, dst = edge_index

        # dst 是目标节点 index
        assert dst.max().item() < num_dst

        # out: 使用 index_add_ 将源节点特征按目标节点索引累加
        # deg: 统计每个目标节点的邻居数量，用于后续平均池化
        deg = torch.zeros((num_dst,), device=device, dtype=z_dst.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=z_dst.dtype))
        weight = 1.0 / torch.sqrt(deg[dst] + 1e-6)

        out = torch.zeros((num_dst, z_dst.size(1)), device=device, dtype=z_dst.dtype)
        # out.index_add_(0, dst, z_src[src])
        out.index_add_(0, dst, z_src[src] * weight.unsqueeze(-1))

        # deg = deg.clamp(min=1.0).unsqueeze(-1)
        # return out / deg
        return out

    def build_struct_repr(self, data, z_sim, z_sem):
        """
        为 microbe / disease 构造结构感知表示
        """
        # SIM 视图结构（相似图）
        mm_edge = data['microbe', 'similar', 'microbe'].edge_index
        dd_edge = data['disease', 'similar', 'disease'].edge_index
        # print(z_sim['microbe'].size(0),z_sim['disease'].size(0))    # 292，39

        s_m_sim = self.neighbor_readout(
            z_sim['microbe'], mm_edge, z_sim['microbe'].size(0)
        )
        s_m_sim = s_m_sim - z_sim['microbe']  # new
        s_d_sim = self.neighbor_readout(
            z_sim['disease'], dd_edge, z_sim['disease'].size(0)
        )
        s_d_sim = s_d_sim - z_sim['disease']  # new

        # SEM 视图结构（异质交互图）
        # microbe ← disease
        s_m_sem = self.cross_neighbor_readout(
            z_sem['disease'],
            z_sem['microbe'],
            data['disease', 'rev_associated', 'microbe'].edge_index,
            z_sem['microbe'].size(0)
        )
        s_m_sem = s_m_sem - z_sem['microbe']  # new

        # disease ← microbe
        s_d_sem = self.cross_neighbor_readout(
            z_sem['microbe'],
            z_sem['disease'],
            data['microbe', 'associated', 'disease'].edge_index,
            z_sem['disease'].size(0)
        )
        s_d_sem = s_d_sem - z_sem['disease']  # new

        return {
            'microbe': (s_m_sim, s_m_sem),
            'disease': (s_d_sim, s_d_sem)
        }

    def struct_cl_loss(self, data, z_sim, z_sem, tau):
        struct = self.build_struct_repr(data, z_sim, z_sem)

        # microbe
        s_m_sim, s_m_sem = struct['microbe']
        s_m_sim = self.cl_loss_m(F.dropout(s_m_sim, p=self.dropout, training=self.training))
        s_m_sem = self.cl_loss_m(F.dropout(s_m_sem, p=self.dropout, training=self.training))
        loss_m = info_nce(s_m_sim, s_m_sem, tau)

        # disease
        s_d_sim, s_d_sem = struct['disease']
        s_d_sim = self.cl_loss_d(F.dropout(s_d_sim, p=self.dropout, training=self.training))
        s_d_sem = self.cl_loss_d(F.dropout(s_d_sem, p=self.dropout, training=self.training))
        loss_d = info_nce(s_d_sim, s_d_sem, tau)

        return loss_m + loss_d


# ==================== 训练和评估 ====================
class Trainer:
    def __init__(self, model, alpha, device='cuda'):
        self.model = model.to(device)
        self.alpha = alpha
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, weight_decay=1e-5
        )

    def train_epoch(self, data, A_train, device, lam_cl, tau1, tau2, epoch):
        """
        A_train : torch.Tensor [Nm,Nd]  训练掩码矩阵（1=正，0=其他）
        """
        self.model.train()
        data = data.to(device)
        A_train = A_train.to(device)
        self.optimizer.zero_grad()

        z_sim, z_sem = self.model.encode(data)
        logits = self.model.decode(z_sim, z_sem)
        cl_loss1 = self.model.ori_cl_loss(z_sim, z_sem, tau1)
        cl_loss2 = self.model.struct_cl_loss(data, z_sim, z_sem, tau2)
        # cl_loss2 = 0

        # ---- 只在训练位置算 BCE ----
        ratio = (A_train.numel() - A_train.sum()) / A_train.sum().clamp(min=1)
        pos_weight = torch.sqrt(ratio)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_bce = loss_fn(logits, A_train)

        if epoch < 50:
            loss = loss_bce + lam_cl * self.alpha * cl_loss1 + lam_cl * (1 - self.alpha) * cl_loss2
        else:
            loss = loss_bce

        loss.backward()
        self.optimizer.step()

        return float(loss.item()), float(loss_bce.item()), float(cl_loss1.item()), float(cl_loss2.item())


# ==================== 数据准备 ====================
def build_A_from_edges(pos_edges: np.ndarray, Nm: int, Nd: int) -> torch.Tensor:
    A = torch.zeros((Nm, Nd), dtype=torch.float32)
    if pos_edges.size > 0:
        A[pos_edges[:, 0], pos_edges[:, 1]] = 1.0
    return A




# ==================== 主程序 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 0. 加载数据
print("=== Loading Data ===")
datasets = ['Disbiome', 'HMDAD']
data_loader = DataLoader(dataset=datasets[1])
mda, edges_md, mm_sim, dd_sim, m_emb, d_emb = data_loader.load_data()

seed = 42
n_splits = 5
val_ratio = 0.1


# 从此处开始，把下面的代码包装在函数中
def train_final_model(
        alpha, lam_cl, tau1, tau2, m_k, d_k,
        seed=42,
        max_epochs=200,
        device=None
):
    set_seed(seed)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_microbes, n_diseases = mda.shape

    graph_builder = GraphBuilder(
        edges_md, mm_sim, dd_sim, m_emb, d_emb,
        m_k=m_k, d_k=d_k
    )

    hetero_data = graph_builder.build_heterogeneous_graph().to(device)

    A_train = build_A_from_edges(edges_md, n_microbes, n_diseases).to(device)


    model = MDAPredictor(
        hetero_data["microbe"].x_sim.size(-1),
        hetero_data["microbe"].x_sem.size(-1),
        hetero_data["disease"].x_sim.size(-1),
        hetero_data["disease"].x_sem.size(-1),
        hidden_dim=256,
        embed_dim=256,
        dropout=0.2,
        num_relations=4
    ).to(device)

    trainer = Trainer(model, alpha, device=device)

    max_epochs = 200
    for epoch in range(max_epochs):
        loss, loss_bce, cl1, cl2 = trainer.train_epoch(
            hetero_data,
            A_train,
            device=device,
            lam_cl=lam_cl,
            tau1=tau1,
            tau2=tau2,
            epoch=epoch
        )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Loss={loss:.4f} | "
                f"BCE={loss_bce:.4f} | "
                f"CL1={cl1:.4f} | "
                f"CL2={cl2:.4f}"
            )

    model.eval()
    with torch.no_grad():
        z_sim, z_sem = model.encode(hetero_data)
        logits = model.decode(z_sim, z_sem)
        score_matrix = torch.sigmoid(logits).cpu().numpy()


    return score_matrix, model


set_seed(seed)

params = {
    "alpha": 0.8,
    "lam_cl": 0.3,
    "tau1": 0.2,
    "tau2": 0.6,
    "m_k": 25,
    "d_k": 20,
}

score_matrix, model = train_final_model(
                            seed=seed,
                            device=device,
                            **params
                        )

torch.save({
    "model_state_dict": model.state_dict(),
}, "case_hmdad.pt")

def load_model_and_predict(
    model_path,
    edges_md,
    mm_sim,
    dd_sim,
    m_emb,
    d_emb,
    m_k,
    d_k,
    device
):
    checkpoint = torch.load(model_path, map_location=device)

    graph_builder = GraphBuilder(
        edges_md, mm_sim, dd_sim, m_emb, d_emb,
        m_k=m_k, d_k=d_k
    )
    hetero_data = graph_builder.build_heterogeneous_graph().to(device)

    model = MDAPredictor(
        hetero_data["microbe"].x_sim.size(-1),
        hetero_data["microbe"].x_sem.size(-1),
        hetero_data["disease"].x_sim.size(-1),
        hetero_data["disease"].x_sem.size(-1),
        hidden_dim=256,
        embed_dim=256,
        dropout=0.2,
        num_relations=4
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        z_sim, z_sem = model.encode(hetero_data)
        logits = model.decode(z_sim, z_sem)
        score_matrix = torch.sigmoid(logits).cpu().numpy()

    return score_matrix

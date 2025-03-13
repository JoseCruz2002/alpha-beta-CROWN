from expl_algos.src.advExOracle.base_oracle import BaseOracle
from tool_API.external_API import abCrown_API
import torch
import numpy as np

class abCrown_Oracle(BaseOracle):

    def __init__(self, costumization_file_name):
        BaseOracle.__init__(self)
        self.oracle = abCrown_API(costumization_file_name)

    def findAdvEx(self, distance, fixed_features, explanation_problem, norm):
        input_trans = explanation_problem.value
        vnnlib = self.get_vnnlib(
            explanation_problem.classification_problem.feature_set,
            fixed_features,
            input_trans   
        )
        #print(f"vnnlib: {vnnlib}")
        input = self.csr_matrix_to_tensor(input_trans)
        verified_status = self.oracle.run_abCrown(input, vnnlib)
        print(f"verified_status by the oracle: {verified_status}")
        if ("unsafe" in verified_status or "sat" in verified_status or
                "attack success" in verified_status):
            return True
        else:
            return False
        
    def get_vnnlib(self, feature_set, fixed_features, input_trans):
        #print(f"input_trans: {input_trans.shape}")
        index_map = {value: idx for idx, value in enumerate(feature_set)}
        fix_idxs = list(index_map[val] for val in fixed_features)
        res = []
        for i in range(len(feature_set)):
            if i in fix_idxs:
                idx_value = 1 if i in input_trans.nonzero()[1] else 0
                res += [[idx_value, idx_value]]
            else:
                res += [[0, 1]]
        return [(np.array(res, dtype=np.float32), [(torch.tensor([[1.0, -1.0, ]]), np.array([0]))])]
            
    def csr_matrix_to_tensor(self, csr_matrix):
        coo_matrix = csr_matrix.tocoo()
        values = coo_matrix.data
        indices = np.vstack((coo_matrix.row, coo_matrix.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo_matrix.shape
        return torch.sparse_coo_tensor(i, v, torch.Size(shape)).to_dense()

        
        

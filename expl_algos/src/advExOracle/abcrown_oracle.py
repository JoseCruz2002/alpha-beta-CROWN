from expl_algos.src.advExOracle.base_oracle import BaseOracle
from tool_API.external_API import abCrown_API
import torch
import numpy as np

class abCrown_Oracle(BaseOracle):

    def __init__(self, distance, norm, label, costumization_file_name,
                 final_rv, benchmark_year=""):
        BaseOracle.__init__(self)
        self.oracle = abCrown_API(costumization_file_name, benchmark_year)
        self.bounds = final_rv[0][0] # benchmark models
        #print(f"self.bounds: {self.bounds}")

    def findAdvEx(self, distance, fixed_features, explanation_problem, norm):
        clf_type = "drebin" if "FFNN" in explanation_problem.classification_problem.classifier else "benchmark"
        if clf_type == "drebin":
            input_trans = explanation_problem.value
            #print(f"input_trans: {input_trans}")
            input = self.csr_matrix_to_tensor(input_trans)
            #print(f"-- abCrown_Oracle.py; findAdvEx; input shape: {input.shape}")
            #print(f"input_trans: {input_trans}")
        else:
            input = torch.tensor([explanation_problem.value])

        vnnlib = self.get_vnnlib(
            feature_set=explanation_problem.classification_problem.feature_set,
            fixed_features=fixed_features,
            distance=distance,
            input=input,
            norm=norm,
            label=explanation_problem.classification
        )
        #print(f"vnnlib: {vnnlib}")
        #print(f"input: {input.shape}; {input[:5]}")
        #print(f"-- abCrown_Oracle.py; findAdvEx; vnnlib[0][0]['X'] shape {vnnlib[0][0]['X'].shape}")
        #print(f"-- abcrown_oracle; findAdvEx; vnnlib: {vnnlib}")
        verified_status = self.oracle.run_abCrown(input, vnnlib)
        print(f"verified_status by the oracle: {verified_status}")
        if (verified_status == "unsafe" or verified_status == "sat" or
                verified_status == "attack success"): #or "unknown" in verified_status):
            return True
        else:
            return False
        
    def get_vnnlib(self, feature_set, fixed_features, distance, input, norm, label):
        index_map = {value: idx for idx, value in enumerate(feature_set)}
        fix_idxs = list(index_map[val] for val in fixed_features)
        res = []
        data_min = []
        data_max = []
        for i in range(len(feature_set)):
            if i in fix_idxs:
                idx_value = input[0][i]
                #res += [[idx_value, idx_value]]
                data_min += [float(idx_value)]
                data_max += [float(idx_value)]
            else:
                #res += [[0, 1]]
                data_min += [self.bounds[i][0]]
                data_max += [self.bounds[i][1]]
        #prop00 = np.array(res, dtype=np.float32)
        prop00 = {
            'X': input,
            'data_min': torch.tensor(data_min),
            'data_max': torch.tensor(data_max),
            'eps': float(distance),
            'norm': norm,
        }
        mat = torch.tensor([[1.0, -1.0, ]]) if label == 1 else torch.tensor([[-1.0, 1.0, ]])
        # Meaning: vnnlib = [prop00, (mat, rhs)]
        return [(prop00, [(mat, torch.tensor([0.0]))])]
        #return [(prop00, [(torch.tensor([[1.0, -1.0]]), np.array([0]))])]


    def csr_matrix_to_tensor(self, csr_matrix):
        coo_matrix = csr_matrix.tocoo()
        values = coo_matrix.data
        indices = np.vstack((coo_matrix.row, coo_matrix.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo_matrix.shape
        return torch.sparse_coo_tensor(i, v, torch.Size(shape)).to_dense()

        
        

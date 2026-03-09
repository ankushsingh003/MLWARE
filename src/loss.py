import torch
import torch.nn as nn
import itertools

class MarginRankingLossPairs(nn.Module):
    def __init__(self, margin=0.1):
        super(MarginRankingLossPairs, self).__init__()
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
        
    def forward(self, scores, targets):
        """
        scores: (B, S) predicted scores for each frame
        targets: (B, S) true indices/ranks for each frame. 
                 Since the labels give the true chronological positions (e.g. [5, 6, 7]),
                 frame A should be ranked higher than frame B if target[A] < target[B].
        """
        B, S = scores.shape
        loss = 0.0
        
        for b in range(B):
            # Generate all pairs (i, j) for this sequence
            pairs = list(itertools.combinations(range(S), 2))
            
            # Tensors to hold pair data
            score_i = []
            score_j = []
            y = []
            
            for i, j in pairs:
                score_i.append(scores[b, i])
                score_j.append(scores[b, j])
                
                # If target[i] < target[j], frame i comes BEFORE frame j chronologically.
                # Let's say we want scores to correspond directly to temporal position.
                # Then score_i should be < score_j.
                # If target[i] < target[j], y = -1 (score_i should be smaller)
                if targets[b, i] < targets[b, j]:
                    y.append(-1.0)
                elif targets[b, i] > targets[b, j]:
                    y.append(1.0)
                else:
                    y.append(0.0)
            
            score_i_tensor = torch.stack(score_i)
            score_j_tensor = torch.stack(score_j)
            y_tensor = torch.tensor(y, device=scores.device).float()
            
            loss += self.loss_fn(score_i_tensor, score_j_tensor, y_tensor)
            
        return loss / B

def calculate_kendall_tau(pred_order, true_order):
    """
    pred_order: list of predicted frame positions
    true_order: list of true frame positions
    Example:
    truth = [2, 3, 4, 1, 0]
    pred = [2, 4, 3, 1, 0]
    Matches Sherlock Files problem statement exactly.
    """
    assert len(pred_order) == len(true_order)
    S = len(pred_order)
    
    # We find concordant and discordant pairs based on relative orders
    # A pair (i, j) is concordant if the relative order of pred_order[i] and pred_order[j]
    # matches the relative order of true_order[i] and true_order[j].
    C = 0
    D = 0
    
    # The true order vector already gives us the relative correctness.
    # To compute Kendall's tau effectively, we map elements to their ranks
    # but the problem statement logic: 
    #   "All frame pairs: (2,3) (2,4) ... 
    #    Correct pairs = 9, Incorrect = 1"
    # To do this programmatically:
    true_pos = {val: idx for idx, val in enumerate(true_order)}
    pred_pos = {val: idx for idx, val in enumerate(pred_order)}
    
    pairs = list(itertools.combinations(true_order, 2))
    
    for (u, v) in pairs:
        # Check relative order in true
        true_diff = true_pos[u] - true_pos[v]
        # Check relative order in pred
        pred_diff = pred_pos[u] - pred_pos[v]
        
        if true_diff * pred_diff > 0:
            C += 1
        elif true_diff * pred_diff < 0:
            D += 1
            
    if C + D == 0:
        return 0.0
    return (C - D) / (C + D)

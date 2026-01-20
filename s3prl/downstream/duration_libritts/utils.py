
import torch

def get_triphone_context(phone_idxs, stress_idxs, intra_word_positions):
    """
    input:
        phone_idxs: list of Tensor, each Tensor is of shape (num_phones,)
        stress_idxs: list of Tensor, each Tensor is of shape (num_phones,)
        intra_word_positions: list of Tensor, each Tensor is of shape (num_phones,)
    returns:
        triphone_phone_idxs: list of Tensor, each Tensor is of shape (num_phones, 3)
        triphone_stress_idxs: list of Tensor, each Tensor is of shape (num_phones, 3)
        triphone_intra_word_positions: list of Tensor, each Tensor is of shape (num_phones, 3)

    triphone_phone_idxs contains the [prev_phone_idx, phone_idx, next_phone_idx] for each phone
    """
    triphone_phone_idxs = []
    triphone_stress_idxs = []
    triphone_intra_word_positions = []
    for p_idxs, s_idxs, iwp_idxs in zip(phone_idxs, stress_idxs, intra_word_positions):
        num_phones = len(p_idxs)
        # use -1 for padding
        triphone_p_idxs = torch.stack([
            torch.cat([torch.tensor([-1], device=p_idxs.device), p_idxs[:-1]]), # previous phone
            p_idxs, # current phone
            torch.cat([p_idxs[1:], torch.tensor([-1], device=p_idxs.device)]) # next phone
        ], dim=-1)
        triphone_phone_idxs.append(triphone_p_idxs)
        triphone_s_idxs = torch.stack([
            torch.cat([torch.tensor([-1], device=s_idxs.device), s_idxs[:-1]]), # previous stress
            s_idxs, # current stress
            torch.cat([s_idxs[1:], torch.tensor([-1], device=s_idxs.device)]) # next stress
        ], dim=-1)
        triphone_stress_idxs.append(triphone_s_idxs)
        triphone_iwp = torch.stack([
            torch.cat([torch.tensor([-1], device=p_idxs.device), iwp_idxs[:-1]]), # previous intra word position
            iwp_idxs, # current intra word position
            torch.cat([iwp_idxs[1:], torch.tensor([-1], device=p_idxs.device)]) # next intra word position
        ], dim=-1)
        triphone_intra_word_positions.append(triphone_iwp)

    return triphone_phone_idxs, triphone_stress_idxs, triphone_intra_word_positions

def get_relative_intra_word_pos(intra_word_positions, word_lengths):
    """
    input:
        intra_word_positions: list of Tensor, each Tensor of shape (num_phones,)
        word_lengths: list of Tensor, each Tensor of shape (num_phones,)
    returns:
        relative_intra_word_pos: list of Tensor, each Tensor of shape (num_phones,)
    """
    relative_intra_word_positions = []
    for iwp, wl in zip(intra_word_positions, word_lengths):
        relative_iwp = iwp.float() / wl.float()
        relative_intra_word_positions.append(relative_iwp)
    return relative_intra_word_positions

class NormalizedSinusoidalEncoding(torch.nn.Module):
    def __init__(self, C):
        super(NormalizedSinusoidalEncoding, self).__init__()
        self.C = C

    def forward(self, pos):
        """
        args:
            pos (B, 3): tensor of normalized positions in [0, 1]
        returns:
            pos_enc (B, 3, C)
        """

        div_term = torch.exp(
            torch.arange(0, self.C, 2, dtype=pos.dtype, device=pos.device) * -(torch.log(torch.tensor(10000.0)) / self.C)
        ) # (C//2,)
        print(f"div_term: {div_term.shape}")
        sinusoid_inp = pos.unsqueeze(-1) * div_term # (B, 3, C//2)
        print(f"sinusoid_inp: {sinusoid_inp.shape}")
        pos_enc = torch.zeros(
            pos.size(0), pos.size(1), self.C, dtype=pos.dtype, device=pos.device,
        )
        print(f"pos_enc: {pos_enc.shape}")
        pos_enc[:, :, 0::2] = torch.sin(sinusoid_inp)
        pos_enc[:, :, 1::2] = torch.cos(sinusoid_inp)

        return pos_enc
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadTemporalAttention(nn.Module):
    """Enhanced multi-head attention specifically for temporal patterns"""
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=200):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
        # Learnable relative positional encodings for temporal patterns
        self.relative_position_embeddings = nn.Parameter(
            torch.randn(2 * max_seq_len - 1, self.head_dim)
        )
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Apply linear transformations
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with relative positions
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add relative positional bias
        rel_pos_bias = self._get_relative_position_bias(seq_len)
        scores = scores + rel_pos_bias
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out(out), attention_weights
    
    def _get_relative_position_bias(self, seq_len):
        """FIXED: Generate relative position bias for temporal attention"""
        device = self.relative_position_embeddings.device
        
        # Create position matrix
        range_vec = torch.arange(seq_len, device=device)
        range_mat = range_vec.repeat(seq_len).view(seq_len, seq_len)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        distance_mat_clipped = torch.clamp(distance_mat, -seq_len + 1, seq_len - 1)
        
        final_mat = distance_mat_clipped + seq_len - 1
        
        # Handle tensor dimensions
        embeddings = self.relative_position_embeddings[final_mat]  # Shape: (seq_len, seq_len, head_dim)
        
        # Expand correctly for multiple heads
        embeddings = embeddings.permute(2, 0, 1)  # Shape: (head_dim, seq_len, seq_len)
        embeddings = embeddings.unsqueeze(0).expand(self.num_heads, -1, -1, -1)  # Shape: (num_heads, head_dim, seq_len, seq_len)
        
        # Return proper shape for attention scores
        return embeddings.mean(dim=1)  # Shape: (num_heads, seq_len, seq_len)

class TemporalConv1DBlocks(nn.Module):
    """Enhanced temporal convolution blocks for emergence pattern detection"""
    def __init__(self, input_dim, d_model, dropout=0.1):
        super().__init__()
        
        # Multi-scale temporal convolutions for different emergence patterns
        self.conv_blocks = nn.ModuleList([
            # Short-term patterns (1-3 hours)
            nn.Sequential(
                nn.Conv1d(input_dim, d_model // 4, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            # Medium-term patterns (3-7 hours) 
            nn.Sequential(
                nn.Conv1d(input_dim, d_model // 4, kernel_size=7, padding=3),
                nn.BatchNorm1d(d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            # Long-term patterns (7-15 hours)
            nn.Sequential(
                nn.Conv1d(input_dim, d_model // 4, kernel_size=15, padding=7),
                nn.BatchNorm1d(d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            # Very long-term patterns (15-31 hours)
            nn.Sequential(
                nn.Conv1d(input_dim, d_model // 4, kernel_size=31, padding=15),
                nn.BatchNorm1d(d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        ])
        
        # Refinement convolution to combine multi-scale features
        self.refinement_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Projection layer to ensure consistent dimensions
        self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        Returns: (batch_size, seq_len, d_model)
        """
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x_conv = x.transpose(1, 2)
        
        # Apply multi-scale convolutions
        conv_outputs = []
        for conv_block in self.conv_blocks:
            conv_out = conv_block(x_conv)
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features
        combined = torch.cat(conv_outputs, dim=1)  # (batch_size, d_model, seq_len)
        
        # Apply refinement convolution
        refined = self.refinement_conv(combined)
        
        # Transpose back: (batch_size, seq_len, d_model)
        output = refined.transpose(1, 2)
        
        # Apply projection for consistent dimensions
        output = self.projection(output)
        
        return output

class EmergencePatternTransformer(nn.Module):
    """
    Enhanced transformer with Conv1D layers for better temporal understanding
    """
    def __init__(self, 
                 input_dim=5,
                 d_model=256,
                 num_heads=8,
                 num_layers=6,
                 dropout=0.1,
                 output_len=12,
                 max_seq_len=200,
                 use_temporal_conv=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_len = output_len
        self.max_seq_len = max_seq_len
        self.use_temporal_conv = use_temporal_conv
        
        # Input processing with optional temporal convolution
        if self.use_temporal_conv:
            print("✅ Using Enhanced Temporal Conv1D layers for emergence pattern detection")
            # Enhanced temporal convolution blocks
            self.temporal_conv = TemporalConv1DBlocks(input_dim, d_model, dropout)
            
            # Additional input projection (optional)
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model)
            )
            
            # Fusion layer to combine conv and linear features
            self.feature_fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            print("❌ Using standard input projection (no Conv1D)")
            # Standard input projection
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model)
            )
        
        # Learnable positional encoding (better than fixed)
        self.positional_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Multi-scale local pattern extractors (existing - kept for compatibility)
        self.local_pattern_extractors = nn.ModuleList([
            nn.Conv1d(d_model, d_model // 4, kernel_size=k, padding=k//2)
            for k in [3, 7, 15, 31]  # Different temporal scales
        ])
        
        # Transformer layers with enhanced attention
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadTemporalAttention(d_model, num_heads, dropout, max_seq_len),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model)
                ),
                'norm2': nn.LayerNorm(d_model)
            })
            for _ in range(num_layers)
        ])
        
        # Emergence-specific prediction head
        self.emergence_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Multi-step prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_len)
        )
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters for better learning"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass with enhanced temporal understanding
        x: (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        if self.use_temporal_conv:
            # Apply enhanced temporal convolution
            conv_features = self.temporal_conv(x)  # (batch, seq_len, d_model)
            
            # Apply standard input projection
            linear_features = self.input_projection(x)  # (batch, seq_len, d_model)
            
            # Fuse conv and linear features
            x = self.feature_fusion(torch.cat([conv_features, linear_features], dim=-1))
        else:
            # Standard input projection
            x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add learnable positional encoding
        if seq_len <= self.max_seq_len:
            x = x + self.positional_embedding[:seq_len].unsqueeze(0)
        
        # Multi-scale local pattern extraction (existing pipeline)
        x_conv = x.transpose(1, 2)  # (batch, d_model, seq_len)
        local_features = []
        for conv in self.local_pattern_extractors:
            local_feat = F.gelu(conv(x_conv))
            local_features.append(local_feat)
        
        # Combine local features
        combined_local = torch.cat(local_features, dim=1)  # (batch, d_model, seq_len)
        combined_local = combined_local.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Residual connection
        x = x + combined_local
        
        # Transformer layers with enhanced attention
        attention_weights = []
        for layer in self.transformer_layers:
            # Multi-head attention with residual
            attended, attn_weights = layer['attention'](x)
            x = layer['norm1'](x + attended)
            
            # Feed-forward with residual
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
            
            if return_attention:
                attention_weights.append(attn_weights)
        
        # Emergence detection (auxiliary task)
        emergence_scores = self.emergence_detector(x)  # (batch, seq_len, 1)
        
        # Weighted pooling based on emergence scores
        emergence_weights = F.softmax(emergence_scores.squeeze(-1), dim=1)  # (batch, seq_len)
        pooled = torch.sum(x * emergence_weights.unsqueeze(-1), dim=1)  # (batch, d_model)
        
        # Final predictions
        predictions = self.prediction_head(pooled)  # (batch, output_len)
        
        if return_attention:
            return predictions, emergence_scores, attention_weights
        
        return predictions

class SARTransformerLocalTile(nn.Module):
    """
    Enhanced wrapper with Conv1D support
    """
    def __init__(self, input_dim=5, d_model=128, nhead=4, num_layers=4, dropout=0.1, output_len=12, max_seq_len=500, use_temporal_conv=True):
        super().__init__()
        
        self.use_temporal_conv = use_temporal_conv
        
        self.transformer = EmergencePatternTransformer(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=nhead,
            num_layers=num_layers,
            dropout=dropout,
            output_len=output_len,
            max_seq_len=max_seq_len,
            use_temporal_conv=use_temporal_conv
        )
        
    def forward(self, x, src_key_padding_mask=None):
        return self.transformer(x)
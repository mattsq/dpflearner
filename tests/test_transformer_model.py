import torch
import pytest
from outdist.models import get_model
from outdist.models.transformer import TransformerDistribution


def test_transformer_forward_shape():
    """Test that transformer model outputs correct shape."""
    model = get_model(
        "transformer",
        in_dim=3,
        start=0.0,
        end=1.0,
        n_bins=4,
        d_model=16,
        n_heads=2,
        n_layers=1,
        pooling="mean",
    )
    x = torch.randn(2, 3)
    logits = model(x)
    assert logits.shape == (2, 4)


def test_transformer_different_pooling_methods():
    """Test that different pooling methods work correctly."""
    pooling_methods = ["mean", "max", "sum", "last"]
    
    for pooling in pooling_methods:
        model = get_model(
            "transformer",
            in_dim=5,
            n_bins=3,
            d_model=8,
            n_heads=2,
            n_layers=1,
            pooling=pooling,
        )
        x = torch.randn(3, 5)
        logits = model(x)
        assert logits.shape == (3, 3)


def test_transformer_different_architectures():
    """Test transformer with different architectural parameters."""
    # Test with different number of layers
    model = get_model(
        "transformer",
        in_dim=2,
        n_bins=5,
        d_model=32,
        n_heads=4,
        n_layers=3,
        dropout=0.2,
    )
    x = torch.randn(4, 2)
    logits = model(x)
    assert logits.shape == (4, 5)
    
    # Test with custom d_ff
    model = get_model(
        "transformer",
        in_dim=2,
        n_bins=5,
        d_model=16,
        n_heads=2,
        n_layers=2,
        d_ff=32,
    )
    x = torch.randn(4, 2)
    logits = model(x)
    assert logits.shape == (4, 5)


def test_transformer_single_feature():
    """Test transformer with single input feature."""
    model = get_model(
        "transformer",
        in_dim=1,
        n_bins=3,
        d_model=8,
        n_heads=2,
        n_layers=1,
    )
    x = torch.randn(5, 1)
    logits = model(x)
    assert logits.shape == (5, 3)


def test_transformer_batch_consistency():
    """Test that transformer produces consistent outputs across batch sizes."""
    model = get_model(
        "transformer",
        in_dim=3,
        n_bins=4,
        d_model=16,
        n_heads=2,
        n_layers=1,
    )
    
    # Test single sample
    x_single = torch.randn(1, 3)
    logits_single = model(x_single)
    assert logits_single.shape == (1, 4)
    
    # Test batch
    x_batch = torch.randn(10, 3)
    logits_batch = model(x_batch)
    assert logits_batch.shape == (10, 4)


def test_transformer_gradient_flow():
    """Test that gradients flow properly through the transformer."""
    model = get_model(
        "transformer",
        in_dim=2,
        n_bins=3,
        d_model=8,
        n_heads=2,
        n_layers=1,
    )
    x = torch.randn(2, 2, requires_grad=True)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # Check that model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_transformer_attention_mask_shapes():
    """Test internal attention computations (implicitly through forward pass)."""
    model = get_model(
        "transformer",
        in_dim=4,
        n_bins=2,
        d_model=12,
        n_heads=3,
        n_layers=2,
    )
    
    # Test with different input sizes
    for batch_size in [1, 5, 10]:
        x = torch.randn(batch_size, 4)
        logits = model(x)
        assert logits.shape == (batch_size, 2)


def test_transformer_invalid_pooling():
    """Test that invalid pooling method raises error."""
    with pytest.raises(ValueError, match="Unknown pooling method"):
        model = get_model(
            "transformer",
            in_dim=2,
            n_bins=3,
            d_model=8,
            n_heads=2,
            n_layers=1,
            pooling="invalid",
        )
        x = torch.randn(1, 2)
        model(x)


def test_transformer_head_dimension_compatibility():
    """Test that d_model is compatible with n_heads."""
    # This should work - d_model divisible by n_heads
    model = get_model(
        "transformer",
        in_dim=2,
        n_bins=3,
        d_model=16,
        n_heads=4,
        n_layers=1,
    )
    x = torch.randn(2, 2)
    logits = model(x)
    assert logits.shape == (2, 3)
    
    # This should fail - d_model not divisible by n_heads
    with pytest.raises(AssertionError):
        model = get_model(
            "transformer",
            in_dim=2,
            n_bins=3,
            d_model=15,  # Not divisible by 4
            n_heads=4,
            n_layers=1,
        )


def test_default_config_instantiates_transformer():
    """Test that default configuration creates valid transformer."""
    cfg = TransformerDistribution.default_config()
    model = get_model(cfg)
    assert isinstance(model, TransformerDistribution)
    
    # Test that it can do forward pass
    x = torch.randn(2, 1)  # Default in_dim is 1
    logits = model(x)
    assert logits.shape == (2, 10)  # Default n_bins is 10


def test_transformer_reproducibility():
    """Test that transformer outputs are reproducible with same seed."""
    torch.manual_seed(42)
    model1 = get_model(
        "transformer",
        in_dim=3,
        n_bins=4,
        d_model=16,
        n_heads=2,
        n_layers=1,
    )
    x = torch.randn(2, 3)
    logits1 = model1(x)
    
    torch.manual_seed(42)
    model2 = get_model(
        "transformer",
        in_dim=3,
        n_bins=4,
        d_model=16,
        n_heads=2,
        n_layers=1,
    )
    logits2 = model2(x)
    
    assert torch.allclose(logits1, logits2, atol=1e-6)
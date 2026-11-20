"""Compare atom and driver-encoded TMA descriptors on SM120."""
import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator
from cutlass.tensor_utils import TensorMapManager, TensorMapUpdateMode


@cute.kernel
def tile_load(atom: cute.CopyAtom, coordinates: cute.Tensor, output: cute.Tensor,
              descriptor: cute.Tensor, use_external_map: cutlass.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    tile, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()
    barrier = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1)).iterator
    shared = smem.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((2, 32), stride=(32, 1)), byte_alignment=128
    )
    source_tile = cute.local_tile(coordinates, (2, 32), (tile, 0))
    shared_view, source_view = cpasync.tma_partition(
        atom, 0, cute.make_layout(1), cute.group_modes(shared, 0, 2),
        cute.group_modes(source_tile, 0, 2),
    )

    with cute.arch.elect_one():
        cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_expect_tx(barrier, 2 * 32 * 2)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    if cutlass.const_expr(use_external_map):
        manager = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
        ptr = manager.get_tensormap_ptr(descriptor.iterator, cute.AddressSpace.generic)
        cute.copy(atom, source_view, shared_view, tma_bar_ptr=barrier, tma_desc_ptr=ptr)
    else:
        cute.copy(atom, source_view, shared_view, tma_bar_ptr=barrier)
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(barrier)
    cute.arch.mbarrier_wait(barrier, 0)
    for row in cutlass.range_constexpr(2):
        output[tile * 2 + row, lane] = shared[row, lane]


@cute.jit
def launch(source: cute.Tensor, output: cute.Tensor, descriptor: cute.Tensor,
           use_external_map: cutlass.Constexpr):
    layout = cute.make_layout((2, 32), stride=(32, 1))
    atom, coordinates = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), source, layout, (2, 32)
    )
    tile_load(atom, coordinates, output, descriptor, use_external_map).launch(
        grid=(2, 1, 1), block=(32, 1, 1)
    )


def encode_map(source, promotion):
    result, tensor_map = driver.cuTensorMapEncodeTiled(
        driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, source.data_ptr(), [driver.cuuint64_t(32), driver.cuuint64_t(source.shape[0])],
        [driver.cuuint64_t(source.stride(0) * source.element_size())],
        [driver.cuuint32_t(32), driver.cuuint32_t(2)], [driver.cuuint32_t(1), driver.cuuint32_t(1)],
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE, promotion,
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {result}")
    words = [int(word) & ((1 << 64) - 1) for word in tensor_map.opaque]
    return torch.tensor(words, dtype=torch.uint64).view(torch.int64).cuda()


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This probe certifies SM120 only")

    storage = (torch.arange(144, device="cuda").reshape(3, 48) + 1).to(torch.bfloat16)
    source = storage[:, :32]
    output = torch.full((4, 32), float("nan"), device="cuda", dtype=torch.bfloat16)
    expected = torch.cat((source, torch.zeros_like(source[:1])))
    promos = driver.CUtensorMapL2promotion
    cases = (
        ("atom", None),
        ("external_no_promotion", promos.CU_TENSOR_MAP_L2_PROMOTION_NONE),
        ("external_128B", promos.CU_TENSOR_MAP_L2_PROMOTION_L2_128B),
        ("external_256B", promos.CU_TENSOR_MAP_L2_PROMOTION_L2_256B),
    )
    placeholder = torch.empty(16, dtype=torch.int64, device="cuda")
    for name, promotion in cases:
        descriptor = placeholder if promotion is None else encode_map(source, promotion)
        views = tuple(from_dlpack(t, assumed_align=16) for t in (source, output, descriptor))
        output.fill_(float("nan"))
        cute.compile(launch, *views, promotion is not None)(*views)
        error = (output.float() - expected.float()).abs().max().item()
        if not torch.equal(output, expected):
            raise AssertionError(f"PROBE FAIL max_diff={error}  # {name}")
        print(f"PROBE PASS max_diff={error}  # {name}")


if __name__ == "__main__":
    main()

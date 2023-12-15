// Copyright (c) 2023, DeepLink.
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include "DIPUStorageImpl.h"


namespace dipu {
DIPUStorageImpl::DIPUStorageImpl(
    use_byte_size_t use_byte_size,
    size_t size_bytes,
    at::Allocator* allocator,
    bool resizable) : c10::StorageImpl(
      use_byte_size,
      size_bytes,
      allocator,
      resizable) {}

void DIPUStorageImpl::release_resources() {
  StorageImpl::release_resources();
}

void DIPUStorageImpl::init_desc(
    c10::IntArrayRef size,
    c10::optional<at::MemoryFormat> memory_format_opt) {
  storage_sizes_.set_sizes(size);
  bool is_contiguous = true;
  if (memory_format_opt.has_value()) {
    is_contiguous = *memory_format_opt == c10::MemoryFormat::Contiguous;
  }
  if (!is_contiguous) {
    format_ = ::diopiMemoryFormat_t::Undefined;
  } else if (size.size() == 5) {
    format_ = diopiMemoryFormat_t::NCDHW;
  } else if (size.size() == 4) {
    format_ = diopiMemoryFormat_t::NCHW;
  } else {
    format_ = diopiMemoryFormat_t::Contiguous;
  }
}

void DIPUStorageImpl::get_desc(diopiStorageDesc_t* desc) const {
  desc->sizes.data = storage_sizes_.sizes_data();
  desc->sizes.len = storage_sizes_.size();
  desc->format = format_;
}

void DIPUStorageImpl::set_desc(const diopiStorageDesc_t& desc) {
  storage_sizes_.set_sizes(c10::IntArrayRef{desc.sizes.data, static_cast<size_t>(desc.sizes.len)});
  format_ = desc.format;
}

bool DIPUStorageImpl::TensorHasStorageDesc(const at::Tensor* tensor) {
  return dynamic_cast<DIPUStorageImpl*>(tensor->storage().unsafeGetStorageImpl()) != nullptr;
}

DIPUStorageImpl* DIPUStorageImpl::GetImplPtr(const at::Tensor& tensor) {
  auto *ptr = dynamic_cast<DIPUStorageImpl*>(tensor.storage().unsafeGetStorageImpl());
  TORCH_CHECK(ptr, "tensor must use DIPUStorageImpl");
  return ptr;
}

DIPUStorageImpl* DIPUStorageImpl::GetImplPtr(const at::Tensor* tensor) {
  auto *ptr = dynamic_cast<DIPUStorageImpl*>(tensor->storage().unsafeGetStorageImpl());
  TORCH_CHECK(ptr, "tensor must use DIPUStorageImpl");
  return ptr;
}

}  // namespace dipu

// Copyright (c) 2023, DeepLink.
#pragma once
#include <csrc_dipu/base/basedef.h>
#include <diopi/diopirt.h>

using dipu::devapis::VendorDeviceType;

namespace dipu {

constexpr const char* VendorTypeToStr(VendorDeviceType t) noexcept {
  switch (t) {
    case VendorDeviceType::MLU:
      return "MLU";
    case VendorDeviceType::CUDA:
      return "CUDA";
    case VendorDeviceType::NPU:
      return "NPU";
    case VendorDeviceType::GCU:
      return "GCU";
    case VendorDeviceType::SUPA:
      return "SUPA";
    case VendorDeviceType::DROPLET:
      return "DROPLET";
  }
  return "null";
}

DIPU_API bool isDeviceTensor(const at::Tensor& tensor);

diopiMemoryFormat_t get_format(const at::Tensor& tensor);

at::Tensor format_cast(at::Tensor tensor, diopiMemoryFormat_t target_format);

DIPU_API bool is_in_bad_fork();
void poison_fork();

}  // namespace dipu

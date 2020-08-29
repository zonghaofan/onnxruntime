// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/arena.h"
#include "core/framework/bfc_arena.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

using DeviceAllocatorFactory = std::function<std::unique_ptr<IDeviceAllocator>(OrtDevice::DeviceId)>;

struct AllocatorCreationInfo {
  AllocatorCreationInfo(DeviceAllocatorFactory device_alloc_factory0,
                        OrtDevice::DeviceId device_id0 = 0,
                        bool use_arena0 = true,
                        OrtArenaCfg arena_cfg0 = {-1, -1, -1, -1})
      : device_alloc_factory(device_alloc_factory0),
        device_id(device_id0),
        use_arena(use_arena0),
        arena_cfg(arena_cfg0) {
  }

  DeviceAllocatorFactory device_alloc_factory;
  OrtDevice::DeviceId device_id;
  bool use_arena;
  OrtArenaCfg arena_cfg;
};

AllocatorPtr CreateAllocator(const AllocatorCreationInfo& info);

}  // namespace onnxruntime

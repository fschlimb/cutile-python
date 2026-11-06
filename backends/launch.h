// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cutile::launch {
namespace nb = nanobind;

struct ManagedTensor {
  nb::dlpack::dltensor tensor;
  void *context;
  void (*deleter)(ManagedTensor *);
};

struct Argument {
  nb::handle scalar;
  nb::object capsule;
  const nb::dlpack::dltensor *tensor = nullptr;
};

struct Slot {
  uint64_t value = 0;
  size_t size = 8;
};

inline int64_t index(nb::handle value) {
  nb::object integer = nb::steal(PyNumber_Index(value.ptr()));
  if (!integer.is_valid())
    throw nb::python_error();
  int64_t result = PyLong_AsLongLong(integer.ptr());
  if (PyErr_Occurred())
    throw nb::python_error();
  return result;
}

inline std::array<int64_t, 3> dimensions(nb::handle grid) {
  std::array<int64_t, 3> result{1, 1, 1};
  if (PyLong_Check(grid.ptr())) {
    result[0] = index(grid);
    return result;
  }
  nb::object sequence = nb::steal(PySequence_Tuple(grid.ptr()));
  if (!sequence.is_valid())
    throw nb::python_error();
  Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence.ptr());
  if (size < 1 || size > 3)
    throw nb::value_error("launch dimensions must have length 1 to 3");
  for (Py_ssize_t dim = 0; dim < size; ++dim)
    result[dim] =
        index(nb::handle(PySequence_Fast_GET_ITEM(sequence.ptr(), dim)));
  return result;
}

inline void synchronize(nb::handle stream) {
  if (stream.is_none())
    return;
  if (PyLong_Check(stream.ptr())) {
    int zero = PyObject_RichCompareBool(stream.ptr(), nb::int_(0).ptr(), Py_EQ);
    if (zero < 0)
      throw nb::python_error();
    if (zero)
      return;
  }
  nb::object function =
      nb::steal(PyObject_GetAttrString(stream.ptr(), "synchronize"));
  if (!function.is_valid()) {
    if (!PyErr_ExceptionMatches(PyExc_AttributeError))
      throw nb::python_error();
    PyErr_Clear();
  }
  if (!function.is_valid() || function.is_none())
    throw nb::type_error("stream must be None, 0, or provide synchronize()");
  function();
}

template <typename T> void append(std::string &key, T value) {
  key.append(reinterpret_cast<const char *>(&value), sizeof(value));
}

template <typename T> Slot slot(T value) {
  Slot result;
  result.size = sizeof(T);
  std::memcpy(&result.value, &value, sizeof(value));
  return result;
}

template <typename T> bool signedSlot(int64_t number, Slot &result) {
  if (number < std::numeric_limits<T>::min() ||
      number > std::numeric_limits<T>::max())
    return false;
  result = slot(static_cast<T>(number));
  return true;
}

template <typename T> bool unsignedSlot(uint64_t number, Slot &result) {
  if (number > std::numeric_limits<T>::max())
    return false;
  result = slot(static_cast<T>(number));
  return true;
}

inline bool scalarSlot(nb::handle value, char format, Slot &result) {
  if (format == '?') {
    result = slot(bool(PyObject_IsTrue(value.ptr())));
    return true;
  }
  if (format == 'f' || format == 'd') {
    double number = PyFloat_AsDouble(value.ptr());
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    if (format == 'f') {
      if (std::isfinite(number) &&
          std::abs(number) > std::numeric_limits<float>::max())
        return false;
      result = slot(static_cast<float>(number));
    } else {
      result = slot(number);
    }
    return true;
  }
  bool isUnsigned =
      format == 'B' || format == 'H' || format == 'I' || format == 'Q';
  if (isUnsigned) {
    uint64_t number = PyLong_AsUnsignedLongLong(value.ptr());
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    switch (format) {
    case 'B': return unsignedSlot<uint8_t>(number, result);
    case 'H': return unsignedSlot<uint16_t>(number, result);
    case 'I': return unsignedSlot<uint32_t>(number, result);
    case 'Q': result = slot(number); return true;
    default: break;
    }
  }
  int64_t number = PyLong_AsLongLong(value.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    return false;
  }
  switch (format) {
  case 'b': return signedSlot<int8_t>(number, result);
  case 'h': return signedSlot<int16_t>(number, result);
  case 'i': return signedSlot<int32_t>(number, result);
  case 'q': result = slot(number); return true;
  default:
    throw nb::value_error("unsupported native scalar format");
  }
}

struct Parameter {
  int rank;
  char format;
};

inline std::vector<Parameter> plan(nb::tuple descriptors) {
  std::vector<Parameter> result;
  for (nb::handle descriptor : descriptors) {
    if (PyTuple_Check(descriptor.ptr())) {
      nb::tuple array = nb::borrow<nb::tuple>(descriptor);
      result.push_back({nb::cast<int>(array[0]), nb::cast<char>(array[1])});
    } else {
      result.push_back({-1, nb::cast<char>(descriptor)});
    }
  }
  return result;
}

template <typename Kernel>
bool pack(const std::vector<Argument> &arguments,
          const std::vector<Parameter> &parameters,
          typename Kernel::Arguments &slots) {
  if (arguments.size() != parameters.size())
    return false;
  auto add = [&](Slot value) {
    if constexpr (Kernel::memref)
      slots.push_back(value);
    else
      slots.push_back(value.value);
  };
  for (size_t arg = 0; arg < arguments.size(); ++arg) {
    const auto &argument = arguments[arg];
    const auto &parameter = parameters[arg];
    if (parameter.rank < 0) {
      Slot value;
      if (argument.tensor ||
          !scalarSlot(argument.scalar, parameter.format, value))
        return false;
      add(value);
      continue;
    }
    if (!argument.tensor || argument.tensor->ndim != parameter.rank)
      return false;
    const auto &tensor = *argument.tensor;
    uintptr_t pointer =
        reinterpret_cast<uintptr_t>(tensor.data) + tensor.byte_offset;
    add(slot(pointer));
    if (Kernel::memref) {
      add(slot(pointer));
      add(slot(int64_t(0)));
    }
    for (const int64_t *dims : {tensor.shape, tensor.strides}) {
      for (int dim = 0; dim < tensor.ndim; ++dim) {
        int64_t value = dims[dim];
        if (value < 0)
          return false;
        if (Kernel::memref || parameter.format == 'q') {
          add(slot(value));
        } else {
          if (value > std::numeric_limits<int32_t>::max())
            return false;
          add(slot(static_cast<int32_t>(value)));
        }
      }
    }
  }
  return true;
}

template <typename Kernel> class Cache {
  struct Entry {
    nb::object loaded;
    std::vector<Parameter> parameters;
    size_t slots = 0;
  };
  struct Cached {
    std::shared_ptr<Entry> entry;
    std::list<std::string>::iterator position;
  };

public:
  Cache(nb::tuple constants, size_t limit) : limit(limit) {
    for (nb::handle constant : constants)
      this->constants.push_back(nb::cast<bool>(constant));
    discoverExporters();
  }

  nb::object launch(nb::tuple args, nb::handle stream, nb::handle grid,
                    nb::handle options) {
    std::string key;
    std::vector<Argument> arguments;
    if (!describe(args, key, arguments))
      return nb::none();
    std::shared_ptr<Entry> entry;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto found = entries.find(key);
      if (found == entries.end())
        return nb::bytes(key.data(), key.size());
      entry = found->second.entry;
      order.splice(order.end(), order, found->second.position);
    }
    typename Kernel::Arguments slots;
    slots.reserve(entry->slots);
    if (!pack<Kernel>(arguments, entry->parameters, slots))
      return nb::none();
    auto dims = dimensions(grid);
    synchronize(stream);
    auto &kernel = nb::cast<Kernel &>(entry->loaded.attr("kernel"));
    kernel.launchNative(slots, dims, options, entry->loaded);
    return nb::bool_(true);
  }

  void insert(nb::tuple args, nb::object loaded, nb::handle expectedKey) {
    std::string key;
    std::vector<Argument> arguments;
    if (!limit || !describe(args, key, arguments))
      return;
    if (!expectedKey.is_none()) {
      nb::bytes expected = nb::cast<nb::bytes>(expectedKey);
      if (key.size() != expected.size() ||
          std::memcmp(key.data(), expected.data(), key.size()) != 0)
        return;
    }
    auto entry =
        std::make_shared<Entry>(Entry{loaded, plan(loaded.attr("native_plan"))});
    for (const auto &parameter : entry->parameters)
      entry->slots += parameter.rank < 0
                          ? 1
                          : 2 * parameter.rank + (Kernel::memref ? 3 : 1);
    std::lock_guard<std::mutex> lock(mutex);
    auto found = entries.find(key);
    if (found != entries.end()) {
      found->second.entry.swap(entry);
      order.splice(order.end(), order, found->second.position);
      return;
    }
    order.push_back(key);
    entries.emplace(std::move(key),
                    Cached{std::move(entry), std::prev(order.end())});
    if (entries.size() > limit) {
      entry = std::move(entries.at(order.front()).entry);
      entries.erase(order.front());
      order.pop_front();
    }
  }

  size_t size() {
    std::lock_guard<std::mutex> lock(mutex);
    return entries.size();
  }

private:
  void discoverExporters() {
    PyObject *modules = PyImport_GetModuleDict();
    PyObject *torchModule = PyDict_GetItemString(modules, "torch");
    if (torchModule && !tensorType.is_valid()) {
      nb::object torch = nb::borrow<nb::object>(torchModule);
      if (nb::hasattr(torch, "Tensor") && nb::hasattr(torch, "_C")) {
        tensorType = torch.attr("Tensor");
        toDlpack = torch.attr("_C").attr("_to_dlpack");
      }
    }
    PyObject *parameterModule =
        PyDict_GetItemString(modules, "torch.nn.parameter");
    if (parameterModule && !parameterType.is_valid()) {
      nb::object module = nb::borrow<nb::object>(parameterModule);
      if (nb::hasattr(module, "Parameter"))
        parameterType = module.attr("Parameter");
    }
    PyObject *numpyModule = PyDict_GetItemString(modules, "numpy");
    if (numpyModule && !numpyArrayType.is_valid()) {
      nb::object module = nb::borrow<nb::object>(numpyModule);
      if (nb::hasattr(module, "ndarray"))
        numpyArrayType = module.attr("ndarray");
    }
  }

  bool describe(nb::tuple args, std::string &key,
                std::vector<Argument> &arguments) {
    if (args.size() != constants.size())
      return false;
    key.reserve(args.size() * 64);
    arguments.reserve(args.size());
    for (size_t arg = 0; arg < args.size(); ++arg) {
      nb::handle value = args[arg];
      uint8_t kind = PyBool_Check(value.ptr())         ? 1
                     : PyLong_CheckExact(value.ptr())  ? 2
                     : PyFloat_CheckExact(value.ptr()) ? 3
                                                       : 4;
      append(key, kind);
      if (constants[arg]) {
        if (kind == 1 || kind == 2) {
          int overflow = 0;
          int64_t number = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
          if (overflow || PyErr_Occurred()) {
            PyErr_Clear();
            return false;
          }
          append(key, number);
        } else if (kind == 3) {
          append(key, PyFloat_AS_DOUBLE(value.ptr()));
        } else {
          return false;
        }
        continue;
      }
      if (kind != 4) {
        arguments.push_back({value, {}, nullptr});
        continue;
      }
      PyObject *valueType = reinterpret_cast<PyObject *>(Py_TYPE(value.ptr()));
      bool torchTensor = tensorType.is_valid() && valueType == tensorType.ptr();
      bool torchParameter = parameterType.is_valid() &&
                            valueType == parameterType.ptr();
      bool numpyArray = numpyArrayType.is_valid() &&
                        valueType == numpyArrayType.ptr();
      if (!torchTensor && !torchParameter && !numpyArray) {
        nb::object typeModule = nb::steal(PyObject_GetAttrString(
            reinterpret_cast<PyObject *>(Py_TYPE(value.ptr())), "__module__"));
        if (!typeModule.is_valid())
          PyErr_Clear();
        const char *moduleName =
            typeModule.is_valid() && PyUnicode_Check(typeModule.ptr())
                ? PyUnicode_AsUTF8(typeModule.ptr())
                : nullptr;
        if (PyErr_Occurred()) {
          PyErr_Clear();
          moduleName = nullptr;
        }
        bool mayBeTorch =
            (!tensorType.is_valid() || !parameterType.is_valid()) &&
          moduleName && (std::strcmp(moduleName, "torch") == 0 ||
                   std::strcmp(moduleName, "torch.nn.parameter") == 0);
        bool mayBeNumpy = !numpyArrayType.is_valid() &&
                          moduleName && std::strcmp(moduleName, "numpy") == 0;
        if (mayBeTorch || mayBeNumpy) {
          discoverExporters();
          torchTensor = tensorType.is_valid() && valueType == tensorType.ptr();
          torchParameter =
              parameterType.is_valid() && valueType == parameterType.ptr();
          numpyArray =
              numpyArrayType.is_valid() && valueType == numpyArrayType.ptr();
        }
      }
      if (!torchTensor && !torchParameter && !numpyArray)
        return false;
      if (!numpyArray && !toDlpack.is_valid())
        return false;
      nb::object capsule = nb::steal(
          numpyArray ? PyObject_CallMethod(value.ptr(), "__dlpack__", "")
                     : PyObject_CallOneArg(toDlpack.ptr(), value.ptr()));
      if (!capsule.is_valid()) {
        PyErr_Clear();
        return false;
      }
      auto *managed = static_cast<ManagedTensor *>(
          PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
      if (!managed) {
        PyErr_Clear();
        return false;
      }
      const auto &tensor = managed->tensor;
      if (tensor.device.device_type != Kernel::device ||
          tensor.device.device_id != 0 || tensor.ndim < 0 ||
          (tensor.ndim && (!tensor.shape || !tensor.strides)))
        return false;
      append(key, tensor.dtype.code);
      append(key, tensor.dtype.bits);
      append(key, tensor.dtype.lanes);
      append(key, tensor.ndim);
      uintptr_t pointer =
          reinterpret_cast<uintptr_t>(tensor.data) + tensor.byte_offset;
      append(key, pointer & -pointer);
      for (int dim = 0; dim < tensor.ndim; ++dim) {
        // DLPack exporters may rewrite strides of singleton dimensions.
        if (tensor.shape[dim] <= 1 || tensor.strides[dim] < 0)
          return false;
        append(key, tensor.shape[dim]);
        append(key, tensor.strides[dim]);
      }
      arguments.push_back({{}, std::move(capsule), &tensor});
    }
    return true;
  }

  size_t limit;
  std::vector<bool> constants;
  nb::object tensorType, parameterType, numpyArrayType, toDlpack;
  std::mutex mutex;
  std::list<std::string> order;
  std::unordered_map<std::string, Cached> entries;
};

template <typename Kernel> void bindCache(nb::module_ &module) {
  using NativeCache = Cache<Kernel>;
  nb::class_<NativeCache>(module, "LaunchCache")
      .def(nb::init<nb::tuple, size_t>())
      .def("launch", &NativeCache::launch, nb::arg("args"),
           nb::arg("stream").none(), nb::arg("grid"), nb::arg("options"))
       .def("insert", &NativeCache::insert, nb::arg("args"), nb::arg("loaded"),
         nb::arg("expected_key").none() = nb::none())
      .def("__len__", &NativeCache::size);
}

} // namespace cutile::launch
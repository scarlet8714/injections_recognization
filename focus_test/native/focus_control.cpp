// Small persistent IAMCameraControl bridge; image capture stays in OpenCV.
#include <Windows.h>
#include <dshow.h>
#include <wrl/client.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

std::string utf8(const wchar_t* value) {
    if (!value || !*value) return {};
    int size = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    std::string result(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), size, nullptr, nullptr);
    result.pop_back();
    return result;
}

std::string quoted(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    out << '"';
    return out.str();
}

void check(HRESULT hr, const char* operation) {
    if (FAILED(hr)) {
        std::ostringstream error;
        error << operation << " failed (HRESULT 0x" << std::hex << std::uppercase
              << static_cast<unsigned long>(hr) << ')';
        throw std::runtime_error(error.str());
    }
}

struct Device {
    ComPtr<IMoniker> moniker;
    std::string name;
    std::string path;
};

std::vector<Device> devices() {
    ComPtr<ICreateDevEnum> enumerator;
    check(CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                           IID_PPV_ARGS(&enumerator)), "Create device enumerator");
    ComPtr<IEnumMoniker> list;
    HRESULT hr = enumerator->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &list, 0);
    check(hr, "Enumerate video devices");
    std::vector<Device> result;
    if (hr == S_FALSE) return result;
    ComPtr<IMoniker> item;
    while (list->Next(1, &item, nullptr) == S_OK) {
        Device device;
        device.moniker = item;
        ComPtr<IPropertyBag> properties;
        if (SUCCEEDED(item->BindToStorage(nullptr, nullptr, IID_PPV_ARGS(&properties)))) {
            for (auto property : {L"FriendlyName", L"DevicePath"}) {
                VARIANT value;
                VariantInit(&value);
                if (SUCCEEDED(properties->Read(property, &value, nullptr)) && value.vt == VT_BSTR) {
                    if (std::wstring(property) == L"FriendlyName") device.name = utf8(value.bstrVal);
                    else device.path = utf8(value.bstrVal);
                }
                VariantClear(&value);
            }
        }
        result.push_back(std::move(device));
        item.Reset();
    }
    return result;
}

class FocusControl {
    ComPtr<IBaseFilter> filter_;
    ComPtr<IAMCameraControl> control_;
    Device device_;
    int index_;
public:
    FocusControl(Device device, int index) : device_(std::move(device)), index_(index) {
        check(device_.moniker->BindToObject(nullptr, nullptr, IID_PPV_ARGS(&filter_)), "Bind camera");
        check(filter_.As(&control_), "Query IAMCameraControl");
    }

    void print() {
        long minimum, maximum, step, default_value, caps, value, flags;
        check(control_->GetRange(CameraControl_Focus, &minimum, &maximum, &step,
                                  &default_value, &caps), "Get focus range");
        check(control_->Get(CameraControl_Focus, &value, &flags), "Get focus value");
        std::cout << "{\"ok\":true,\"index\":" << index_ << ",\"name\":" << quoted(device_.name)
                  << ",\"device_path\":" << quoted(device_.path)
                  << ",\"minimum\":" << minimum << ",\"maximum\":" << maximum
                  << ",\"step\":" << step << ",\"default\":" << default_value
                  << ",\"capabilities\":" << caps << ",\"value\":" << value
                  << ",\"flags\":" << flags << "}" << std::endl;
    }

    void set(long value, long flags) {
        long minimum, maximum, step, default_value, caps;
        check(control_->GetRange(CameraControl_Focus, &minimum, &maximum, &step,
                                  &default_value, &caps), "Get focus range");
        if (flags != CameraControl_Flags_Manual && flags != CameraControl_Flags_Auto)
            throw std::runtime_error("Focus flag must be manual=2 or auto=1");
        if (!(caps & flags)) throw std::runtime_error("Requested focus mode is unsupported");
        if (flags == CameraControl_Flags_Manual &&
            (value < minimum || value > maximum || step <= 0 || (value - minimum) % step))
            throw std::runtime_error("Focus value is outside the reported range or step");
        check(control_->Set(CameraControl_Focus, value, flags), "Set focus");
        print();
    }
};

void error(const std::exception& e) {
    std::cout << "{\"ok\":false,\"error\":" << quoted(e.what()) << "}" << std::endl;
}

int main(int argc, char** argv) {
    HRESULT initialized = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(initialized)) return 2;
    int code = 0;
    try {
        auto available = devices();
        if (argc > 1 && std::string(argv[1]) == "--list") {
            std::cout << "{\"ok\":true,\"devices\":[";
            for (size_t i = 0; i < available.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << "{\"index\":" << i << ",\"name\":" << quoted(available[i].name)
                          << ",\"device_path\":" << quoted(available[i].path) << '}';
            }
            std::cout << "]}" << std::endl;
        } else {
            int index = argc > 1 ? std::stoi(argv[1]) : 0;
            if (index < 0 || index >= static_cast<int>(available.size()))
                throw std::runtime_error("Camera index not found");
            FocusControl camera(std::move(available[index]), index);
            camera.print();
            if (!(argc > 2 && std::string(argv[2]) == "--probe")) {
                for (std::string line; std::getline(std::cin, line);) {
                    try {
                        std::istringstream request(line);
                        std::string command;
                        request >> command;
                        if (command == "QUIT") break;
                        if (command == "GET") camera.print();
                        else if (command == "SET") {
                            long value, flags;
                            if (!(request >> value >> flags)) throw std::runtime_error("Expected SET value flags");
                            camera.set(value, flags);
                        } else throw std::runtime_error("Unknown command");
                    } catch (const std::exception& e) { error(e); }
                }
            }
        }
    } catch (const std::exception& e) { error(e); code = 1; }
    CoUninitialize();
    return code;
}

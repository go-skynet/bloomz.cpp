package bloomz

// #cgo LDFLAGS: -lbloomz -lm -lstdc++ -L./
// #cgo darwin LDFLAGS: -framework Accelerate
// #cgo darwin CXXFLAGS: -std=c++11
// #include <bloomz.h>
import "C"
import (
	"fmt"
	"strings"
	"unsafe"
)

type Bloomz struct {
	state unsafe.Pointer
}

func New(model string, opts ...ModelOption) (*Bloomz, error) {
	mo := NewModelOptions(opts...)
	state := C.bloomz_allocate_state()
	modelPath := C.CString(model)
	result := C.bloomz_bootstrap(modelPath, state, C.int(mo.ContextSize), C.bool(mo.F16Memory))
	if result != 0 {
		return nil, fmt.Errorf("failed loading model")
	}

	return &Bloomz{state: state}, nil
}
func (l *Bloomz) Free() {
	C.bloomz_free_model(l.state)
}

func (l *Bloomz) Predict(text string, opts ...PredictOption) (string, error) {

	po := NewPredictOptions(opts...)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, po.Tokens)

	params := C.bloomz_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat))
	ret := C.bloomz_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")

	C.bloomz_free_params(params)

	return res, nil
}

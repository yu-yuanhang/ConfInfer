#ifndef __RESHAPE_H_CA__
#define __RESHAPE_H_CA__

#include <All.h>
#include <core/Layer.h>
#include <core/Param.h>

namespace Kernel {
namespace core {

class Flatten;

class Flatten_L:
virtual public Layer
{
friend class Flatten;
protected:
    Flatten_L() = delete;
    ~Flatten_L();
    Flatten_L(INT startDim,
              INT endDim,
              OpSignature *opSignature);

protected:
    void makeParams(Params *params) override;
    void makeOutputs() override;
    UINT calcWorkspaceSize() override;

public:
    INT startDim() const { return _startDim; }
    INT endDim() const { return _endDim; }

private:
    INT _startDim;
    INT _endDim;
};

class Flatten:
virtual public OpSignature {
public:
    Flatten() = delete;
    ~Flatten();
    explicit Flatten(INT start_dim = 1,
                     INT end_dim = -1);

    Layer &operator()() = delete;
    Layer &operator()(Value_t &value);

private:
    INT _startDim;
    INT _endDim;
};

} // namespace core
} // namespace Kernel

#endif

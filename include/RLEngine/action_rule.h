#pragma once
#include "include.h"
namespace tensorFly{
    namespace action{
        class Action{
            virtual void run();
        };

        class TvmAction:public Action{

        };

    }
}
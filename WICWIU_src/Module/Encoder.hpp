#ifndef __ENCODER__
#define __ENCODER__    value

#include "../Module.hpp"


template<typename DTYPE> class Encoder : public Module<DTYPE>{
private:

    int timesize;

public:

    Encoder(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputsize, hiddensize, use_bias, pName);
    }


    virtual ~Encoder() {}

    int Alloc(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int use_bias, std::string pName) {

        timesize = pInput->GetResult()->GetTimeSize();
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        //------------------------------weight 생성-------------------------
        Tensorholder<DTYPE> *pWeight_x2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "RecurrentLayer_pWeight_x2h_" + pName);
        //Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);
        Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::IdentityMatrix(1, 1, 1, hiddensize, hiddensize), "RecurrentLayer_pWeight_h2h_" + pName);

        Tensorholder<DTYPE> *rBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_" + pName);

        //embedding 추가???
        //out = new Embedding<DTYPE>(pWeight_in, out, "embedding");

        out = new SeqRecurrent<DTYPE>(out, pWeight_x2h, pWeight_h2h, rBias);

        this->AnalyzeGraph(out);

        return TRUE;
    }

    //m_numOfExcutableOperator 이게 private로 되어있어서!!! 그래서 접근이 불가능!!!
    int ForwardPropagate(int pTime) {

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for(int ti=0; ti<timesize; ti++){
            for (int i = 0; i < numOfExcutableOperator; i++) {
                (*ExcutableOperator)[i]->ForwardPropagate(pTime);
            }
        }
        return TRUE;
    }

    int BackPropagate(int pTime) {

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for(int ti=0; ti<timesize; ti++){
            for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
                (*ExcutableOperator)[i]->BackPropagate(pTime);
            }
        }
        return TRUE;
    }

};


#endif  // __ENCODER__

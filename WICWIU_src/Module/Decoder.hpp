#ifndef __DECODER__
#define __DECODER__    value

#include "../Module.hpp"


template<typename DTYPE> class Decoder : public Module<DTYPE>{
private:

    int timesize;

    Operator<DTYPE> *m_initHiddenTensorholder;
    Tensor<DTYPE> * m_initHidden;

public:

    Decoder(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, int inputsize, int hiddensize, int outputsize, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, pEncoder, inputsize, hiddensize, outputsize, use_bias, pName);
    }


    virtual ~Decoder() {}


    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, int inputsize, int hiddensize, int outputsize, int use_bias, std::string pName) {

        timesize = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();


        //*************************************************************************hidden 값 가져오기!!!****************************************************************************
        m_initHiddenTensorholder  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, batchsize, 1, 1, hiddensize), "tempHidden");
        Tensor<DTYPE> *m_initHidden = Tensor<DTYPE>::Zeros(1, batchsize, 1, 1, hiddensize);

/*
이런 함수가 module에 존재!!! 따라서 아래와같이 getResult를 사용해서 접근가능!!!
        template<typename DTYPE> Tensor<DTYPE> *Module<DTYPE>::GetResult() const {
            return m_pLastOperator->GetResult();
        }
*/

/*중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요

alloc에서는 값을 복사해서 넘겨줘봤자... 소용이 없지...
alloc은 이제.... 한번만 호출되고 끝나니깐....!!! 그러니깐... 값을 복사해서 넘겨주는 부분은 forward에 존재해야함!!!
매 time마다는 아니지만.. 학습마다 실행되어야 하는 부분!!!
*/

        Tensor<DTYPE> *_initHidden = pEncoder->GetResult();                     //이렇게 접근하는거 가능!!! 이유는 위에 함수때문에 가능!!!
        Shape *_initShape = _initHidden->GetShape();
        Shape *initShape = m_initHidden->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for(int i=0; i<hiddensize; i++){
                (*m_initHidden)[Index5D(initShape, 0, ba, 0, 0, i)] = (*_initHidden)[Index5D(_initShape, timesize-1, ba, 0, 0, i)];
            }
        }

        //만약에 Tensorholder로 해서 operator형으로 넘겨주고 싶은거면!
        m_initHiddenTensorholder  = new Tensorholder<DTYPE>(m_initHidden, "tempHidden");
        //*********************************************************************************여기까지가 hidden값 가져오기!!!************************************************************

        this->SetInput(2, pInput, pEncoder);           //여기 Encoder도 같이 연결해줌!!!

        Operator<DTYPE> *out = pInput;

        //pEncoder        ????

        //------------------------------weight 생성-------------------------
        Tensorholder<DTYPE> *pWeight_x2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "RecurrentLayer_pWeight_x2h_" + pName);
        //Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);
        Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::IdentityMatrix(1, 1, 1, hiddensize, hiddensize), "RecurrentLayer_pWeight_h2h_" + pName);

        Tensorholder<DTYPE> *pWeight_h2o = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, outputsize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_h2o_" + pName);

        Tensorholder<DTYPE> *rBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_" + pName);

        out = new SeqRecurrent<DTYPE>(out, pWeight_x2h, pWeight_h2h, rBias, m_initHiddenTensorholder);                           //tensor 넘겨주는지 operator 넘겨주는지 이걸로ㄱㄱ!!!

        out = new MatMul<DTYPE>(pWeight_h2o, out, "rnn_matmul_ho");

        if (use_bias) {
            Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, outputsize, 0.f), "Add_Bias_" + pName);
            out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        }

        this->AnalyzeGraph(out);

        return TRUE;
    }

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

/*
        template<typename DTYPE> Tensor<DTYPE> *Module<DTYPE>::GetGradient() const {
            return m_pLastOperator->GetGradient();
        }
        이런 함수가 존재해서 이제 GetGradient로 접근가능!!!
*/
        //Encoder로 넘겨주기!!!
        //encoder의 마지막 time에만 넘겨주면됨!!!
        Tensor<DTYPE> *enGradient = this->GetInput()[1]->GetGradient();
        Tensor<DTYPE> *_enGradient = m_initHiddenTensorholder->GetGradient();

        Shape *enShape  = enGradient->GetShape();
        Shape *_enShape = _enGradient->GetShape();

        int enTimesize = enGradient->GetTimeSize();
        int batchSize = enGradient->GetBatchSize();
        int colSize = enGradient->GetColSize();

        //encoder의 Gradient에 저장해주기!!!  굳이 hidden에 저장해주지 않아도 되는게 hidden2ouptut은 rnn에 없어서! 옆에서 누나 위에서 주나 똑같음!!
        for(int ba=0; ba < batchSize; ba++){
            for(int col=0; col < colSize; col++){
                (*enGradient)[Index5D(enShape, enTimesize-1, ba, 0, 0, col)] = (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, col)];
            }
        }


        return TRUE;
    }

};


#endif  // __DECODER__

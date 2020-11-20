#include "net/my_RNN.hpp"
#include <time.h>
#include <unistd.h>

#define BATCH                 1
#define EPOCH                 2
#define MAX_TRAIN_ITERATION    20000   // (60000 / BATCH)
#define MAX_TEST_ITERATION     1   // (10000 / BATCH)
#define GPUID                 1



int main(int argc, char const *argv[]) {
    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;

/*
    char alphabet[] = "hi$";
        int outDim = 2;

        char inputStr[] = "hi";
        char desiredStr[] = "i$";

        char seqLen = 2;
        int inputSeq[] = { 0, 1, 1, 0 };
    	int desiredSeq[] = { 0, 1, 1, 1 };

        Tensor<float> *x_tensor = new Tensor<float>(seqLen, BATCH, 1, 1, 2);

        Tensorholder<float> *x_holder = new Tensorholder<float>(seqLen, BATCH, 1, 1, 2, "x(input)");

        Tensor<float> *label_tensor = new Tensor<float>(seqLen, BATCH, 1, 1, 2);
        Tensorholder<float> *label_holder = new Tensorholder<float>(seqLen, BATCH, 1, 1, 2, "label");

        NeuralNetwork<float> *net = new my_RNN(x_holder,label_holder);

        for(int t = 0; t < seqLen; t++){
            for (int ba = 0; ba < 1; ba++) {
                for(int col = 0; col < 2; col++){
                    (*x_tensor)[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)] = inputSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
                    (*label_tensor)[Index5D(label_tensor->GetShape(), t, ba, 0, 0, col)] = desiredSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
                }
            }
        }
*/


    char alphabet[] = "helo";
    int outDim = 4;

    char inputStr[] = "hell";
    char desiredStr[] = "ello";

    char seqLen = 4;
    int inputSeq[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0 };
	  int desiredSeq[] = { 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

    Tensor<float> *x_tensor = new Tensor<float>(4, BATCH, 1, 1, 4);
    //Tensorholder<float> *x_holder = new Tensorholder<float>(x_tensor, "x");                             //get rank에서 문제가 생기는 거는 이걸 바꿔주니깐 해결됨
    Tensorholder<float> *x_holder = new Tensorholder<float>(4, BATCH, 1, 1, 4, "x");

    Tensor<float> *label_tensor = new Tensor<float>(4, BATCH, 1, 1, 4);
    //Tensorholder<float> *label_holder = new Tensorholder<float>(label_tensor, "label");
    Tensorholder<float> *label_holder = new Tensorholder<float>(4, BATCH, 1, 1, 4, "x");

    NeuralNetwork<float> *net = new my_RNN(x_holder,label_holder);

    for(int t = 0; t < seqLen; t++){
        for (int ba = 0; ba < 1; ba++) {
            for(int col = 0; col < 4; col++){
                (*x_tensor)[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)] = inputSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
                (*label_tensor)[Index5D(label_tensor->GetShape(), t, ba, 0, 0, col)] = desiredSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
            }
        }
    }



/*
char alphabet[] = "heloab";
    int outDim = 6;

    char inputStr[] = "helloa";
    char desiredStr[] = "elloab";

    char seqLen = 6;
    int inputSeq[] = { 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0};
	int desiredSeq[] = { 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1 };

    Tensor<float> *x_tensor = new Tensor<float>(6, BATCH, 1, 1, 6);
    //Tensorholder<float> *x_holder = new Tensorholder<float>(x_tensor, "x");
    Tensorholder<float> *x_holder = new Tensorholder<float>(6, BATCH, 1, 1, 6, "x(input)");

    Tensor<float> *label_tensor = new Tensor<float>(6, BATCH, 1, 1, 6);
    //Tensorholder<float> *label_holder = new Tensorholder<float>(label_tensor, "label");
    Tensorholder<float> *label_holder = new Tensorholder<float>(6, BATCH, 1, 1, 6, "label");

    // Tensorholder<float> *x = new Tensorholder<float>(4, BATCH, 1, 1, 4, "x");
    // Tensorholder<float> *label = new Tensorholder<float>(4, BATCH, 1, 1, 4, "label");
    // Tensor<float> *x_t = new Tensor<DTYPE>(1, BATCH, 1, 1, 4);
    // Tensor<float> *l_t = new Tensor<DTYPE>(1, BATCH, 1, 1, 4);

    NeuralNetwork<float> *net = new my_RNN(x_holder,label_holder);

    for(int t = 0; t < seqLen; t++){
        for (int ba = 0; ba < 1; ba++) {
            for(int col = 0; col < 6; col++){
                (*x_tensor)[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)] = inputSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
                (*label_tensor)[Index5D(label_tensor->GetShape(), t, ba, 0, 0, col)] = desiredSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
            }
        }
    }
*/


    std::cout<<'\n';
    net->PrintGraphInformation();



    float best_acc = 0;
    int   epoch    = 0;

    net->FeedInputTensor(2, x_tensor, label_tensor);

    for (int i = epoch + 1; i < EPOCH; i++) {

        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        net->SetModeTrain();

        startTime = clock();

        //net->FeedInputTensor(2, x_tensor, label_tensor);                        //왜??? 왜 안에 넣어두면 안되는거지???

        // ============================== Train ===============================
        std::cout << "Start Train" <<'\n';

        for (int j = 0; j < MAX_TRAIN_ITERATION; j++) {


            // std::cin >> temp;
            //net->FeedInputTensor(2, x_tensor, label_tensor);                     //이 부분이 MNIST에서는 dataloader로 가져가서 이렇게 for문 안에 넣어둠
            net->ResetParameterGradient();
            net->TimeTrain(seqLen);

            // std::cin >> temp;
            //train_accuracy += net->GetAccuracy(4);                               // default로는 10으로 되어있음   이게 기존꺼임
            //train_avg_loss += net->GetLoss();

            train_accuracy = net->GetAccuracy(seqLen);
            train_avg_loss = net->GetLoss();

            std::cout<<" 전달해준 loss값 : "<<net->GetLoss()<<'\n';

            printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f"  ,
                   j + 1, MAX_TRAIN_ITERATION,
                   train_avg_loss, ///  (j + 1),                              //+=이니깐 j+1로 나눠주는거는 알겠는데........ 근데 왜 출력되는 값이 계속 작아지는 거지??? loss값이 같아도 왜 이건 작아지는거냐고...
                   train_accuracy  /// (j + 1)
                 );
            //std::cout<<'\n';
            fflush(stdout);

        }

        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ======================= Test ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInference();

        std::cout << "Start Test" <<'\n';

        //net->FeedInputTensor(2, x_tensor, label_tensor);

        for (int j = 0; j < (int)MAX_TEST_ITERATION; j++) {
            // #ifdef __CUDNN__
            //         x_t->SetDeviceGPU(GPUID);
            //         l_t->SetDeviceGPU(GPUID);
            // #endif  // __CUDNN__

            //net->FeedInputTensor(2, x_tensor, label_tensor);
            net->TimeTest(seqLen);

            test_accuracy += net->GetAccuracy(seqLen);
            test_avg_loss += net->GetLoss();

            printf("\rTest complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, MAX_TEST_ITERATION,
                   test_avg_loss / (j + 1),
                   test_accuracy / (j + 1));
            fflush(stdout);
        }

        std::cout << "\n\n";

    }       // 여기까지가 epoc for문

    delete net;

    return 0;
}



/*
int main(int argc, char const *argv[]) {


  char alphabet[] = "hi$";
      int outDim = 2;

      char inputStr[] = "hi";
      char desiredStr[] = "i$";

      char seqLen = 2;
      int inputSeq[] = { 0, 1, 1, 0 };
  	int desiredSeq[] = { 0, 1, 1, 1 };

      Tensor<float> *x_tensor = new Tensor<float>(seqLen, BATCH, 1, 1, 2);

      Tensorholder<float> *x_holder = new Tensorholder<float>(seqLen, BATCH, 1, 1, 2, "x(input)");

      Tensor<float> *label_tensor = new Tensor<float>(seqLen, BATCH, 1, 1, 2);
      Tensorholder<float> *label_holder = new Tensorholder<float>(seqLen, BATCH, 1, 1, 2, "label");

      NeuralNetwork<float> *net = new my_RNN(x_holder,label_holder);

      for(int t = 0; t < seqLen; t++){
          for (int ba = 0; ba < 1; ba++) {
              for(int col = 0; col < 2; col++){
                  (*x_tensor)[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)] = inputSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
                  (*label_tensor)[Index5D(label_tensor->GetShape(), t, ba, 0, 0, col)] = desiredSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
              }
          }
      }


    net->PrintGraphInformation();


    float best_acc = 0;
    int   epoch    = 0;

    net->FeedInputTensor(2, x_tensor, label_tensor);

    for (int i = epoch + 1; i < EPOCH; i++) {



        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        net->SetModeTrain();


        // ============================== Train ===============================
        std::cout << "====================Start Train" <<'\n';

        for (int j = 0; j < MAX_TRAIN_ITERATION; j++) {

            //usleep(000);

            net->ResetParameterGradient();
            net->TimeTrain(seqLen);



            train_accuracy = net->GetAccuracy(seqLen);
            train_avg_loss = net->GetLoss();

            //std::cout<<" 전달해준 loss값 : "<<net->GetLoss()<<'\n';

            printf("Train complete percentage is %d / %d -> loss : %f, acc : %f"  ,
                   j + 1, MAX_TRAIN_ITERATION,
                   train_avg_loss, ///  (j + 1),
                   train_accuracy // / (j + 1)
                   );
                   std::cout<<'\n';
            //fflush(stdout);
        }
        std::cout<<'\n';



        // ======================= Test ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInference();

        std::cout << "======================Start Test" <<'\n';


        for (int j = 0; j < (int)MAX_TEST_ITERATION; j++) {

            net->TimeTest(seqLen);

            test_accuracy += net->GetAccuracy(seqLen);
            test_avg_loss += net->GetLoss();

            printf("\rTest complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, MAX_TEST_ITERATION,
                   test_avg_loss / (j + 1),
                   test_accuracy / (j + 1));
            fflush(stdout);
        }

        std::cout << "\n\n";

    }

    delete net;

    return 0;
}
*/

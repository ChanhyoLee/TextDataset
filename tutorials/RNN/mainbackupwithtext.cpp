#include "net/my_RNN.hpp"
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <fstream>   // ifstream 이게 파일 입력
#include <cstring>    //strlen 때문에 추가한 해더
#include <algorithm> //sort 때문에 추가한 헤더

using namespace std;

#define BATCH                 1
#define EPOCH                 2
#define MAX_TRAIN_ITERATION    20000   // (60000 / BATCH)
#define MAX_TEST_ITERATION     1   // (10000 / BATCH)
#define GPUID                 1


int char2index(char* vocab, char c);
char index2char(char* vocab, int index);
void makeonehotvector(int* onehotvector, int vocab_size, int index);

int main(int argc, char const *argv[]) {
    clock_t startTime = 0, endTime = 0;
    double  nProcessExcuteTime = 0;


    //파일 처리하는 부분 추가
    ifstream read;

    int File_length = 0;   //파일 길이 저장
    char *buffer;     //파일의 내용을 저장
    //string buffer;
    char* vocab = new char[100];       //고유 문자 모음
    int flag = 0 ;     //고유 문자를 찾기위한 flag 중복이면 1, 처음보는거면 0
    int vocab_length = 0;

    //read.open("Data/test.txt");
    read.open("Data/middlesize.txt");
    //read.open("Data/shakespeare.txt");

    if(read.is_open()){

      //파일 사이즈 구하기
      read.seekg(0, ios::end);
      File_length = read.tellg();
      read.seekg(0, ios::beg);        //포인터를 다시 시작위치로 바꿈

      //파일 길이만큼 할당
      buffer = new char[File_length];

      //파일 읽기
      read.read(buffer, File_length);

      read.close();

    }
    //고유문자 찾아서 vocab에 넣기
    for(int i=0; i<File_length; i++){

        flag = 0;
        vocab_length = (int)strlen(vocab);

        //std::cout<<"확인하는 문자 : "<<buffer[i]<<'\n';
        //std::cout<<"vocab_length : "<<vocab_length<<'\n';

        for(int j=0; j<vocab_length; j++){
            if(vocab[j]==buffer[i])
              flag = 1;
            }

        if(flag==0){
          vocab[vocab_length] = buffer[i];
        }
    }
    sort(vocab, vocab+strlen(vocab));


    //파일 잘 읽었는지 확인하기
    std::cout<<"파일의 길이 :"<<File_length<<'\n';
    std::cout<<"vocab 길이 : "<<strlen(vocab)<<'\n';

    //vocab 다 출력해보기
    //for(int i=0; i<strlen(vocab); i++)
    //    std::cout<<i<<"번째 단어 : "<<vocab[i]<<'\n';



    Tensorholder<float> *x_holder = new Tensorholder<float>(File_length, BATCH, 1, 1, strlen(vocab), "x");
    Tensorholder<float> *label_holder = new Tensorholder<float>(File_length, BATCH, 1, 1, strlen(vocab), "label");

    NeuralNetwork<float> *net = new my_RNN(x_holder,label_holder);


    int* onehotvector = new int[strlen(vocab)];


    Tensor<float> *x = new Tensor<float>(File_length, 1, 1, 1, strlen(vocab));
    Tensor<float> *label = new Tensor<float>(File_length, 1, 1, 1, strlen(vocab));

    //input data 만들기
    for(int i=0; i<File_length; i++){
    //  std::cout<<i<<"번째 단어 "<<buffer[i]<<" input data값 : "<<'\n';
        for(int j=0; j<strlen(vocab); j++){
            makeonehotvector(onehotvector, strlen(vocab), char2index(vocab, buffer[i]));
            (*x)[Index5D(x->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
  //          std::cout<<(*x)[Index5D(x->GetShape(), i, 0, 0, 0, j)];
        }
    //  std::cout<<'\n';
    }

    //label 만들기
    //1 부터 시작!!!
    //여기도 수정 필요
    for(int i=0; i<File_length; i++){

        //맨 마지막 label 처리
        if(i==File_length-1){
          //std::cout<<"마지막 단어 처리"<<'\n';
          for(int j=0; j<strlen(vocab); j++){
              (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = 0;
          }
          continue;
        }

      //  std::cout<<i<<"번째 단어 "<<buffer[i+1]<<" label data값 : "<<'\n';
        for(int j=0; j<strlen(vocab); j++){
            makeonehotvector(onehotvector, strlen(vocab), char2index(vocab, buffer[i+1]));
            (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
      //      std::cout<<(*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)];
        }
    //    std::cout<<'\n';
    }


    //tensor값 확인해보기
    //std::cout<<"tensor의 값을 확인해보자"<<'\n';
    //std::cout<<"입력 tensor의 값 :"<<x;
    //std::cout<<"label tensor의 값 :"<<label;

    std::cout<<'\n';
    net->PrintGraphInformation();



    float best_acc = 0;
    int   epoch    = 0;

    net->FeedInputTensor(2, x, label);

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
            net->TimeTrain(File_length);

            // std::cin >> temp;
            //train_accuracy += net->GetAccuracy(4);                               // default로는 10으로 되어있음   이게 기존꺼임
            //train_avg_loss += net->GetLoss();


            train_accuracy = net->GetAccuracy(strlen(vocab));
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
            net->TimeTest(File_length);

            test_accuracy += net->GetAccuracy(strlen(vocab));
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



int char2index(char* vocab, char c){

    for(int index=0; index<strlen(vocab); index++){
        if(vocab[index]==c)
          return index;
    }

    std::cout<<"해당 문자는 vocab에 없습니다."<<'\n';
    exit(0);
}

char index2char(char* vocab, int index){

    return vocab[index];
}

void makeonehotvector(int* onehotvector, int vocab_size, int index){

    for(int i=0; i<vocab_size; i++){
        if(i==index)
            onehotvector[i] = 1;
        else
            onehotvector[i] = 0;
    }
}

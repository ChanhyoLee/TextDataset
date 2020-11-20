#include <iostream>
#include "Field_utils.hpp"

using namespace std;


class Field {
private:
  bool sequential;          // 시퀀셜
  string* preprocessing;
  // bool use_vocab;
  // bool init_token;
  // bool eos_token;
  // bool fix_length;
  // string postprocessing;
  // bool lower;
  // bool include_lengths;


  // bool tokenize;        // 뭐지?
  // tokenizer_language='en';
  //batch_first=False,
  // pad_token="<pad>",
  // unk_token="<unk>",
  // pad_first=False,
  // truncate_first=False,
  // stop_words=None,
public:
  Field(bool psequential, string delimiters = ",.?!\"\'><:-") {

    sequential = psequential;

    if(delimiters != "\0")
      preprocessing = new string(delimiters);
  }

  void makeSequential() {
    if(sequential==false) {
      cout<<"Sequential Option is Off"<<endl;
      return;
    }

    field_sequential();
  }

  // void makePreprocess() {
  //   if(sequential == NULL) {
  //     cout<<"Sequential Option is Off"<<endl;
  //     return;
  //   }
  //
  //
  // }

};

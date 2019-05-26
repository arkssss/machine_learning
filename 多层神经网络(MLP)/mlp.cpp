/************************************************************************
*  mlp.cpp - Implements a multi-layer back-propagation neural network
*  CSCI964/CSCI464 3-Layer MLP (Input layer, one hidden layer, and output layer)
*  Ver1: Koren Ward - 15 March 2003
*  Ver2: Koren Ward - 21 July  2003 - Dynamic memory added
*  Ver3: Koren Ward - 20 March 2005 - Net paramaters in datafile added
*  Ver4: FangZhou -   03 April 2019 - 3-, 4- & 5-layer mlp & test fn added
*  Copyright - University of Wollongong - 2005
*************************************************************************/
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<ctime>

using namespace std;

const int MAXN = 50;       // Max neurons in any layer
const int MAXPATS = 5000;  // Max training patterns

// mlp paramaters
long  NumIts ;     // Max training iterations
int   NumHN  ;     // Number of hidden layers
int   NumHN1 ;     // Number of neurons in hidden layer 1
int   NumHN2 ;     // Number of neurons in hidden layer 2
int   NumHN3 ;     // Number of neurons in hidden layer 3
int   NumHN4 ;     // Number of neurons in hidden layer 4
int   Normalizing; // add Normalizing to control if Normalizing the data
float LrnRate;     // Learning rate
float Mtm1   ;     // Momentum(t-1)
float Mtm2   ;     // Momentum(t-2)
float ObjErr ;     // Objective error

// mlp weights
float **w1,**w11,**w111;// wts from input layer to the hidden layer;
float **w_h1, **w_h11, **w_h111 ;// wts from hidden layer1 to the hiddent layer2
float **w_h2, **w_h22, **w_h222 ;// wts from hidden layer1 to the hiddent layer2
float **w2,**w22,**w222;// wts from the hidden layer to output layer;

void TrainNet3(float **x,float **d,int NumIPs,int NumOPs,int NumPats, int Ordering);
void TrainNet4(float **x,float **d,int NumIPs,int NumOPs,int NumPats, int Ordering);
void TrainNet5(float **x,float **d,int NumIPs,int NumOPs,int NumPats, int Ordering);
void TestNet3(float **x,float **d,int NumIPs,int NumOPs,int NumPats);
void TestNet4(float **x,float **d,int NumIPs,int NumOPs,int NumPats);
void TestNet5(float **x,float **d,int NumIPs,int NumOPs,int NumPats);
void DrawGraph(int NumIPs,int NumOPs);
float **Aloc2DAry(int m,int n);
void Free2DAry(float **Ary2D,int n);

int main(){
  ifstream fin;
    
  // my add code
    
  // add Ordering to control the order of data
  int Ordering;
  
  // my add acode
    
  int i,j,NumIPs,NumOPs,NumTrnPats,NumTstPats;
  char Line[500],Tmp[20],FName[20];
  cout<<"Enter data filename: ";
  cin>>FName; cin.ignore();
  fin.open(FName);
  if(!fin.good()){cout<<"File not found!\n";exit(1);}
  //read data specs...
  do{fin.getline(Line,500);}while(Line[0]==';'); //eat comments
  sscanf(Line,"%s%d",Tmp,&NumIPs);
  fin>>Tmp>>NumOPs;
  fin>>Tmp>>NumTrnPats;
  fin>>Tmp>>NumTstPats;
  fin>>Tmp>>NumIts;
  fin>>Tmp>>NumHN;
  i=NumHN;
  // code for read how many the layers the net going to be ...
  if(i-- > 0)fin>>Tmp>>NumHN1;
  if(i-- > 0)fin>>Tmp>>NumHN2;
  if(i-- > 0)fin>>Tmp>>NumHN3;
  if(i-- > 0)fin>>Tmp>>NumHN4;
  fin>>Tmp>>LrnRate;
  fin>>Tmp>>Mtm1;
  fin>>Tmp>>Mtm2;
  fin>>Tmp>>ObjErr;
  fin>>Tmp>>Ordering;      // Ordering for changing training pattern's order
  fin>>Tmp>>Normalizing;   // Normalizing to control the data wether needs to be Normalized
  if( NumIPs<1||NumIPs>MAXN||NumOPs<1||NumOPs>MAXN||
		NumTrnPats<1||NumTrnPats>MAXPATS||NumTrnPats<1||NumTrnPats>MAXPATS||
      NumIts<1||NumIts>20e6||NumHN1<0||NumHN1>50||
      LrnRate<0||LrnRate>1||Mtm1<0||Mtm1>10||Mtm2<0||Mtm2>10||ObjErr<0||ObjErr>10
    ){ cout<<"Invalid specs in data file!\n"; exit(1); }
  float **IPTrnData= Aloc2DAry(NumTrnPats,NumIPs);
  float **OPTrnData= Aloc2DAry(NumTrnPats,NumOPs);
  float **IPTstData= Aloc2DAry(NumTstPats,NumIPs);
  float **OPTstData= Aloc2DAry(NumTstPats,NumOPs);
    
  //add my code
  // store the max and min number of the patterns for Normalizing
  float *NormalizingMax = new float[NumIPs];
  for(i=0;i<NumIPs;i++) NormalizingMax[i]=-3.4e38;
  float *NormalizingMin = new float[NumIPs];
  for(i=0;i<NumIPs;i++) NormalizingMin[i]=3.4e38;
  //add my code
    
  for(i=0;i<NumTrnPats;i++){
	 for(j=0;j<NumIPs;j++)
     {
         fin>>IPTrnData[i][j];
         //add code for collectting the min and max feature number
         NormalizingMax[j] = IPTrnData[i][j] > NormalizingMax[j] ? IPTrnData[i][j] : NormalizingMax[j];
         NormalizingMin[j] = IPTrnData[i][j] < NormalizingMin[j] ? IPTrnData[i][j] : NormalizingMin[j];
         //add code for collectting the min and max feature number
     }
	 for(j=0;j<NumOPs;j++)
		fin>>OPTrnData[i][j];
  }
  for(i=0;i<NumTstPats;i++){
	 for(j=0;j<NumIPs;j++)
		fin>>IPTstData[i][j];
	 for(j=0;j<NumOPs;j++)
		fin>>OPTstData[i][j];
  }

  fin.close();
    
  
  //add code for normalizing the data
  if(Normalizing){
    // data need to be Normalized
    float *gap = new float[NumIPs];
      for(i=0;i<NumIPs;i++) gap[i] = NormalizingMax[i] - NormalizingMin[i];
      for(i=0;i<NumTrnPats;i++){
          for(j=0;j<NumIPs;j++){
              //Normalized
              IPTrnData[i][j] = (IPTrnData[i][j] - NormalizingMin[j]) / gap[j] ;
          }
      }
  }
  //add code for normalizing the data
    
  // add code for select trainNet and testNet
  switch (NumHN) {
      case 1:
           // TrainNet and TestNet for 3 layer
           TrainNet3(IPTrnData,OPTrnData,NumIPs,NumOPs,NumTrnPats,Ordering);
           TestNet3(IPTstData,OPTstData,NumIPs,NumOPs,NumTstPats);
          break;
      case 2:
          // TrainNet and TestNet for 4 layer
           TrainNet4(IPTrnData,OPTrnData,NumIPs,NumOPs,NumTrnPats,Ordering);
           TestNet4(IPTstData,OPTstData,NumIPs,NumOPs,NumTstPats);
          break;
      case 3:
          // TrainNet and TestNet for 5 layer
           TrainNet5(IPTrnData,OPTrnData,NumIPs,NumOPs,NumTrnPats,Ordering);
           TestNet5(IPTstData,OPTstData,NumIPs,NumOPs,NumTstPats);
          break;
      default:
          cout<< "TrainNet choose error";
          exit(1);
         break;
    }
    
  //  DrawGraph(NumIPs,NumOPs); // dram a two-D grapg (for data1)
  // add code for select trainNet and testNet
    
//  TestNet(IPTstData,OPTstData,NumIPs,NumOPs,NumTstPats);
  Free2DAry(IPTrnData,NumTrnPats);
  Free2DAry(OPTrnData,NumTrnPats);
  Free2DAry(IPTstData,NumTstPats);
  Free2DAry(OPTstData,NumTstPats);
  cout<<"End of program.\n";
  system("PAUSE");
  return 0;
}
/**
 init TrainNet with 1 hidden layer
 */
void TrainNet3(float **x,float **d,int NumIPs,int NumOPs,int NumPats, int Ordering){
// Trains 3-layer back propagation neural network
// x[][]=>input data, d[][]=>desired output data
  
  float *h1 = new float[NumHN1]; // O/Ps of hidden layer
  float *y  = new float[NumOPs]; // O/P of Net
  float *ad1= new float[NumHN1]; // HN1 back prop errors
  float *ad2= new float[NumOPs]; // O/P back prop errors
  float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
  int p,i,j;     // for loops indexes
  long ItCnt=0;  // Iteration counter
  long NumErr=0; // Error counter (added for spiral problem)
    
    
  //  ---------------  my add order code
  int *indexArray = new int[NumPats];       // this is order index array to control the order of x
  for (int kk=0; kk < NumPats; kk++) indexArray[kk] = kk;  // initalize the indexArray to give order
    switch (Ordering) {
        case 0 :
            // case 0 : Always uses the same given order of training patterns.
            // case 1 : Make completely different (random) order at each epoch (i.e. new permutation). (change in the loop)
            break;
        case 1 : case 2:
            // case 2 : Random permutation at start.
            srand((unsigned)time(0));
            for (int kk=0; kk< NumPats; kk++){
                int RandomNumber = rand() % NumPats; // 0 到 NumPats -1 的随机数
                int Tmp = indexArray[kk];
                //Swap kk <--> RandomNumber
                indexArray[kk] = indexArray[RandomNumber];
                indexArray[RandomNumber] = Tmp;
            }
            break;
        default:
            break;
    }
  //  ---------------  my add order code
    
  
    
  cout<<"TrainNet3: IP:"<<NumIPs<<" H1:"<<NumHN1<<" OP:"<<NumOPs<<endl;
  cout<<"Params: LrnRate: " << LrnRate << " Mtm1: " << Mtm1 << " Mtm2: "<< Mtm2<<endl;
  cout<<"Configure: Ordering: " << Ordering <<  " IsNormalizing: " << boolalpha << (bool)Normalizing <<endl;
  cout.setf(ios::fixed|ios::showpoint);
  cout<<setprecision(6)<<setw(6)<<"   "<<setw(12)<<"MinErr"<<setw(12)<<"AveErr"<<setw(12)<<"MaxErr"<<setw(12)<<"%error"<<endl;
  // Allocate memory for weights
  w1   = Aloc2DAry(NumIPs,NumHN1);// wts from input layer to the hidden layer;
  w11  = Aloc2DAry(NumIPs,NumHN1);
  w111 = Aloc2DAry(NumIPs,NumHN1);
  w2   = Aloc2DAry(NumHN1,NumOPs);// wts from the hidden layer to output layer;
  w22  = Aloc2DAry(NumHN1,NumOPs);
  w222 = Aloc2DAry(NumHN1,NumOPs);

  // Init wts between -0.5 and +0.5
  srand(time(0));
  for(i=0;i<NumIPs;i++)
    for(j=0;j<NumHN1;j++)
    w1[i][j]=w11[i][j]=w111[i][j]= float(rand())/RAND_MAX - 0.5;
  for(i=0;i<NumHN1;i++)
    for(j=0;j<NumOPs;j++)
      w2[i][j]=w22[i][j]=w222[i][j]= float(rand())/RAND_MAX - 0.5;

  for(;;){// Main learning loop
    MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;
      
      
      //  ---------------  my add order code
//      if(Ordering == 1){
//
//          srand((unsigned)time(0));
//          for (int kk=0; kk< NumPats; kk++){
//              int RandomNumber = rand() % NumPats; // 0 到 NumPats -1 的随机数
//              int Tmp = indexArray[kk];
//              //Swap kk <--> RandomNumber
//              indexArray[kk] = indexArray[RandomNumber];
//              indexArray[RandomNumber] = Tmp;
//          }
//
//      }
      
      if(Ordering == 2) {
          //After each epoch, two patterns are randomly selected and exchanged.
          int RandomNumber1 = rand() % NumPats; // 0 到 NumPats -1 的随机数
          int RandomNumber2 = rand() % NumPats;
          int Tmp = indexArray[RandomNumber1];
          //Swap RandomNumber1 <--> RandomNumber2
          indexArray[RandomNumber1] = indexArray[RandomNumber2];
          indexArray[RandomNumber2] = Tmp;
      }
      //  ---------------  my add order code
      
      
      
    for(p=0;p<NumPats;p++){ // for each pattern... update every step
      p = indexArray[p] ; // using an index array to rearrange the selection order.
      // Cal neural network output
      for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
        float in=0;
          for(j=0;j<NumIPs;j++){
//           cout<< j << "," <<  i << w1[j][i] << " ";
          in+=w1[j][i]*x[p][j];
          }
        h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
      }
//        cout<< endl;
      for(i=0;i<NumOPs;i++){ // Cal O/P of output layer
        float in=0;
        for(j=0;j<NumHN1;j++){
//            cout<< j << "," <<  i << w2[j][i] << " ";
          in+=w2[j][i]*h1[j];
        }
        y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
      }
      // Cal error for this pattern
      PatErr=0.0;
      for(i=0;i<NumOPs;i++){
//          cout<<y[i] << ": " << d[p][i] << " ";
        float err=y[i]-d[p][i]; // actual-desired O/P
        if(err>0)PatErr+=err; else PatErr-=err;
        NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
      }
      if(PatErr<MinErr)MinErr=PatErr;
      if(PatErr>MaxErr)MaxErr=PatErr;
      AveErr+=PatErr;

      float tmp;
      // Learn pattern with back propagation
      for(i=0;i<NumOPs;i++){ // Modify wts from the hidden layer to output layer;
        ad2[i]=(d[p][i]-y[i])*y[i]*(1.0-y[i]);
        for(j=0;j<NumHN1;j++){
          tmp = w2[j][i];
            w2[j][i]+=LrnRate*h1[j]*ad2[i] +
                    Mtm1*(w2[j][i]-w22[j][i]) +
                    Mtm2*(w22[j][i]-w222[j][i]);
          w222[j][i]=w22[j][i];
          w22[j][i]=tmp;
        }
      }
      for(i=0;i<NumHN1;i++){ // Modify wts from input layer to the hidden layer;
        float err=0.0;
        for(j=0;j<NumOPs;j++)
          err+=ad2[j]*w22[i][j];
        ad1[i]=err*h1[i]*(1.0-h1[i]);
        for(j=0;j<NumIPs;j++){
          tmp = w1[j][i];
            w1[j][i]+=LrnRate*x[p][j]*ad1[i] +
                    Mtm1*(w1[j][i]-w11[j][i]) +
                    Mtm2*(w11[j][i]-w111[j][i]);
          w111[j][i]=w11[j][i];
          w11[j][i]=tmp;
        }
      }
    }// end for each pattern
    ItCnt++;
    AveErr/=NumPats;
    float PcntErr = NumErr/float(NumPats) * 100.0;
    cout.setf(ios::fixed|ios::showpoint);
    cout<<setprecision(6)<<setw(6)<<ItCnt<<": "<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<<PcntErr<<endl;

    if((AveErr<=ObjErr)||(ItCnt==NumIts)) break;
  }// end main learning loop
  // Free memory
  delete[] h1; delete[] y;
  delete[] ad1; delete[] ad2;
  delete[] indexArray;
}


/**
 MLP with 4-layers
 2-hidden layer
 */
void TrainNet4(float **x,float **d,int NumIPs,int NumOPs,int NumPats, int Ordering){
    // Trains 3-layer back propagation neural network
    // x[][]=>input data, d[][]=>desired output data
//    ofstream fin;
//    fin.open("error.txt");
    
    float *h1 = new float[NumHN1]; // O/Ps of hidden layer1
    float *h2 = new float[NumHN2]; // O/Ps of hidden layer2
    float *y  = new float[NumOPs]; // O/P of Net
    float *ad1= new float[NumHN1]; // HN1 back prop errors
    float *ad_h1 = new float[NumHN2]; // HN2 - HN1 back prop errors
    float *ad2= new float[NumOPs]; // O/P back prop errors
    float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
    int p,i,j;     // for loops indexes
    long ItCnt=0;  // Iteration counter
    long NumErr=0; // Error counter (added for spiral problem)
    
    
    //  ---------------  my add order code
    int *indexArray = new int[NumPats];       // this is order index array to control the order of x
    for (int kk=0; kk < NumPats; kk++) indexArray[kk] = kk;  // initalize the indexArray to give order
    switch (Ordering) {
        case 0:
            // Always uses the same given order of training patterns.
            break;
        case 1: case 2:
            // case 1 : Make completely different (random) order at each epoch (i.e. new permutation).
            // case 2 : Random permutation at start.
            srand((unsigned)time(0));
            for (int kk=0; kk< NumPats; kk++){
                int RandomNumber = rand() % NumPats; // 0 到 NumPats -1 的随机数
                int Tmp = indexArray[kk];
                //Swap kk <--> RandomNumber
                indexArray[kk] = indexArray[RandomNumber];
                indexArray[RandomNumber] = Tmp;
            }
            break;
        default:
            break;
    }
    //  ---------------  my add order code
    
    
    
    cout<<"TrainNet4: IP:"<<NumIPs<<" H1:"<<NumHN1<<" H2: " <<NumHN2 << " OP:"<<NumOPs<<endl;
    cout<<"Params: LrnRate: " << LrnRate << " Mtm1: " << Mtm1 << " Mtm2: "<< Mtm2<<endl;
    cout<<"Configure: Ordering: " << Ordering <<  " IsNormalizing: " << boolalpha << (bool)Normalizing <<endl;
    cout.setf(ios::fixed|ios::showpoint);
    cout<<setprecision(6)<<setw(6)<<"   "<<setw(12)<<"MinErr"<<setw(12)<<"AveErr"<<setw(12)<<"MaxErr"<<setw(12)<<"%error"<<endl;
    
    // Allocate memory for weights
    w1   = Aloc2DAry(NumIPs,NumHN1);// wts from input layer to the hidden layer;
    w11  = Aloc2DAry(NumIPs,NumHN1);
    w111 = Aloc2DAry(NumIPs,NumHN1);
    
    w_h1     = Aloc2DAry(NumHN1,NumHN2); // wts from hidden layer1 to the hidden layer2;
    w_h11    = Aloc2DAry(NumHN1,NumHN2);
    w_h111   = Aloc2DAry(NumHN1,NumHN2);
    
    w2   = Aloc2DAry(NumHN2,NumOPs);// wts from the hidden layer to output layer;
    w22  = Aloc2DAry(NumHN2,NumOPs);
    w222 = Aloc2DAry(NumHN2,NumOPs);
    
    // Init wts between -0.5 and +0.5
    srand(time(0));
    for(i=0;i<NumIPs;i++)
        for(j=0;j<NumHN1;j++)
            w1[i][j]=w11[i][j]=w111[i][j]= float(rand())/RAND_MAX - 0.5;
    
    // initalize the hidden 1 to hidden 2
    for (i=0;i<NumHN1;i++)
        for(j=0;j<NumHN2;j++)
            w_h1[i][j] = w_h11[i][j] = w_h111[i][j] = float(rand())/RAND_MAX - 0.5;
    
    for(i=0;i<NumHN2;i++)
        for(j=0;j<NumOPs;j++)
            w2[i][j]=w22[i][j]=w222[i][j]= float(rand())/RAND_MAX - 0.5;
    
    for(;;){// Main learning loop
        MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;
        
        //  ---------------  my add order code
        
        //      if(Ordering == 1){
        //
        //          srand((unsigned)time(0));
        //          for (int kk=0; kk< NumPats; kk++){
        //              int RandomNumber = rand() % NumPats; // 0 到 NumPats -1 的随机数
        //              int Tmp = indexArray[kk];
        //              //Swap kk <--> RandomNumber
        //              indexArray[kk] = indexArray[RandomNumber];
        //              indexArray[RandomNumber] = Tmp;
        //          }
        //
        //      }
        
        if(Ordering == 2) {
            //After each epoch, two patterns are randomly selected and exchanged.
            int RandomNumber1 = rand() % NumPats; // 0 到 NumPats -1 的随机数
            int RandomNumber2 = rand() % NumPats;
            int Tmp = indexArray[RandomNumber1];
            //Swap RandomNumber1 <--> RandomNumber2
            indexArray[RandomNumber1] = indexArray[RandomNumber2];
            indexArray[RandomNumber2] = Tmp;
        }
        //  ---------------  my add order code
        
        
        for(p=0;p<NumPats;p++){ // for each pattern... update every step
            
            p = indexArray[p] ; // using an index array to rearrange the selection order.
            
            // Cal neural network output
            for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
                float in=0;
                for(j=0;j<NumIPs;j++){
                    //           cout<< j << "," <<  i << w1[j][i] << " ";
                    in+=w1[j][i]*x[p][j];
                }
                h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
            }
            
            
            for(i=0; i<NumHN2;i++) { // Cal O/P of hidden layer 2
                float in=0;
                for(j=0;j<NumHN1;j++){
                    in+=w_h1[j][i]*h1[j];
                }
                h2[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
            }
            
            
            for(i=0;i<NumOPs;i++){ // Cal O/P of output layer
                float in=0;
                for(j=0;j<NumHN2;j++){
                    //            cout<< j << "," <<  i << w2[j][i] << " ";
                    in+=w2[j][i]*h2[j];
                }
                y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
            }
            // Cal error for this pattern
            PatErr=0.0;
            for(i=0;i<NumOPs;i++){
                //          cout<<y[i] << ": " << d[p][i] << " ";
                float err=y[i]-d[p][i]; // actual-desired O/P
                if(err>0)PatErr+=err; else PatErr-=err;
                NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
            }
            if(PatErr<MinErr)MinErr=PatErr;
            if(PatErr>MaxErr)MaxErr=PatErr;
            AveErr+=PatErr;
            
            float tmp;
            // Learn pattern with back propagation
            for(i=0;i<NumOPs;i++){ // Modify wts from the hidden layer to output layer;
                ad2[i]=(d[p][i]-y[i])*y[i]*(1.0-y[i]);
                for(j=0;j<NumHN2;j++){
                    tmp = w2[j][i];
                    w2[j][i]+=LrnRate*h2[j]*ad2[i] +
                    Mtm1*(w2[j][i]-w22[j][i]) +
                    Mtm2*(w22[j][i]-w222[j][i]);
                    w222[j][i]=w22[j][i];
                    w22[j][i]=tmp;
                }
            }
            
            for(i=0;i<NumHN2;i++){ // Modify wts from the hidden layer2 to hidden layer1
                float err=0.0;
                for(j=0;j<NumOPs;j++)   //error
                    err+=ad2[j]*w22[i][j];
                ad_h1[i]=err*h2[i]*(1.0-h2[i]);
                for(j=0;j<NumHN1;j++){
                    tmp = w_h1[j][i];
                    w_h1[j][i]+=LrnRate*h1[j]*ad_h1[i] +
                    Mtm1*(w_h1[j][i]-w_h11[j][i]) +
                    Mtm2*(w_h11[j][i]-w_h111[j][i]);
                    w_h111[j][i]=w_h11[j][i];
                    w_h11[j][i]=tmp;
                }
            }
            
            for(i=0;i<NumHN1;i++){ // Modify wts from input layer to the hidden layer;
                float err=0.0;
                for(j=0;j<NumHN2;j++)
                    err+=ad_h1[j]*w_h11[i][j];
                ad1[i]=err*h1[i]*(1.0-h1[i]);
                for(j=0;j<NumIPs;j++){
                    tmp = w1[j][i];
                    w1[j][i]+=LrnRate*x[p][j]*ad1[i] +
                    Mtm1*(w1[j][i]-w11[j][i]) +
                    Mtm2*(w11[j][i]-w111[j][i]);
                    w111[j][i]=w11[j][i];
                    w11[j][i]=tmp;
                }
            }
        }// end for each pattern
        ItCnt++;
        AveErr/=NumPats;
        float PcntErr = NumErr/float(NumPats) * 100.0;
//        fin << PcntErr << endl;
        cout.setf(ios::fixed|ios::showpoint);
        cout<<setprecision(6)<<setw(6)<<ItCnt<<": "<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<<PcntErr<<endl;
        
        if((AveErr<=ObjErr)||(ItCnt==NumIts)) break;
    }// end main learning loop
    // Free memory
//    fin.close();
    delete[] h1; delete[] y;
    delete[] ad1; delete[] ad2;
    delete[] indexArray;
}

/**
 MLP with 5-layers
 3-hidden layer
 */
void TrainNet5(float **x,float **d,int NumIPs,int NumOPs,int NumPats, int Ordering){
    // Trains 3-layer back propagation neural network
    // x[][]=>input data, d[][]=>desired output data
//    ofstream fin;
//    fin.open("error.txt");
    
    float *h1 = new float[NumHN1]; // O/Ps of hidden layer1
    float *h2 = new float[NumHN2]; // O/Ps of hidden layer2
    float *h3 = new float[NumHN3]; // O/Ps of hidden layer2
    float *y  = new float[NumOPs]; // O/P of Net
    float *ad1= new float[NumHN1]; // HN1 back prop errors
    float *ad_h1 = new float[NumHN2]; // HN2 - HN1 back prop errors
    float *ad_h2 = new float[NumHN3]; // HN3 - HN2 back prop errors
    float *ad2= new float[NumOPs]; // O/P back prop errors
    float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
    int p,i,j;     // for loops indexes
    long ItCnt=0;  // Iteration counter
    long NumErr=0; // Error counter (added for spiral problem)
    
    
    //  ---------------  my add order code
    int *indexArray = new int[NumPats];       // this is order index array to control the order of x
    for (int kk=0; kk < NumPats; kk++) indexArray[kk] = kk;  // initalize the indexArray to give order
    switch (Ordering) {
        case 0:
            // Always uses the same given order of training patterns.
            break;
        case 1: case 2:
            // case 1 : Make completely different (random) order at each epoch (i.e. new permutation).
            // case 2 : Random permutation at start.
            srand((unsigned)time(0));
            for (int kk=0; kk< NumPats; kk++){
                int RandomNumber = rand() % NumPats; // 0 到 NumPats -1 的随机数
                int Tmp = indexArray[kk];
                //Swap kk <--> RandomNumber
                indexArray[kk] = indexArray[RandomNumber];
                indexArray[RandomNumber] = Tmp;
            }
            break;
        default:
            break;
    }
    //  ---------------  my add order code
    
    
    cout<<"TrainNet5: IP:"<<NumIPs<<" H1:"<<NumHN1<<" H2: " <<NumHN2 << " H3: "<< NumHN3 << " OP:"<<NumOPs<<endl;
    cout<<"Params: LrnRate: " << LrnRate << " Mtm1: " << Mtm1 << " Mtm2: "<< Mtm2<<endl;
    cout<<"Configure: Ordering: " << Ordering <<  " IsNormalizing: " << boolalpha << (bool)Normalizing <<endl;
    cout.setf(ios::fixed|ios::showpoint);
    cout<<setprecision(6)<<setw(6)<<"   "<<setw(12)<<"MinErr"<<setw(12)<<"AveErr"<<setw(12)<<"MaxErr"<<setw(12)<<"%error"<<endl;
    
    // Allocate memory for weights
    w1   = Aloc2DAry(NumIPs,NumHN1);// wts from input layer to the hidden layer;
    w11  = Aloc2DAry(NumIPs,NumHN1);
    w111 = Aloc2DAry(NumIPs,NumHN1);
    
    w_h1     = Aloc2DAry(NumHN1,NumHN2); // wts from hidden layer1 to the hidden layer2;
    w_h11    = Aloc2DAry(NumHN1,NumHN2);
    w_h111   = Aloc2DAry(NumHN1,NumHN2);
    
    w_h2     = Aloc2DAry(NumHN2,NumHN3); // wts from hidden layer1 to the hidden layer3;
    w_h22    = Aloc2DAry(NumHN2,NumHN3);
    w_h222   = Aloc2DAry(NumHN2,NumHN3);
    
    w2   = Aloc2DAry(NumHN3,NumOPs);// wts from the hidden layer to output layer;
    w22  = Aloc2DAry(NumHN3,NumOPs);
    w222 = Aloc2DAry(NumHN3,NumOPs);
    
    // Init wts between -0.5 and +0.5
    srand(time(0));
    for(i=0;i<NumIPs;i++)
        for(j=0;j<NumHN1;j++)
            w1[i][j]=w11[i][j]=w111[i][j]= float(rand())/RAND_MAX - 0.5;
    
    // initalize the hidden 1 to hidden 2
    for (i=0;i<NumHN1;i++)
        for(j=0;j<NumHN2;j++)
            w_h1[i][j] = w_h11[i][j] = w_h111[i][j] = float(rand())/RAND_MAX - 0.5;
    
    // initalize the hidden 2 to hidden 3
    for (i=0;i<NumHN2;i++)
        for(j=0;j<NumHN3;j++)
            w_h2[i][j] = w_h22[i][j] = w_h222[i][j] = float(rand())/RAND_MAX - 0.5;
    
    
    for(i=0;i<NumHN3;i++)
        for(j=0;j<NumOPs;j++)
            w2[i][j]=w22[i][j]=w222[i][j]= float(rand())/RAND_MAX - 0.5;
    
    for(;;){// Main learning loop
        MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;
        
        //  ---------------  my add order code
        
        //      if(Ordering == 1){
        //
        //          srand((unsigned)time(0));
        //          for (int kk=0; kk< NumPats; kk++){
        //              int RandomNumber = rand() % NumPats; // 0 到 NumPats -1 的随机数
        //              int Tmp = indexArray[kk];
        //              //Swap kk <--> RandomNumber
        //              indexArray[kk] = indexArray[RandomNumber];
        //              indexArray[RandomNumber] = Tmp;
        //          }
        //
        //      }
        
        if(Ordering == 2) {
            //After each epoch, two patterns are randomly selected and exchanged.
            int RandomNumber1 = rand() % NumPats; // 0 到 NumPats -1 的随机数
            int RandomNumber2 = rand() % NumPats;
            int Tmp = indexArray[RandomNumber1];
            //Swap RandomNumber1 <--> RandomNumber2
            indexArray[RandomNumber1] = indexArray[RandomNumber2];
            indexArray[RandomNumber2] = Tmp;
        }
        //  ---------------  my add order code
        
        
        for(p=0;p<NumPats;p++){ // for each pattern... update every step
            
            p = indexArray[p] ; // using an index array to rearrange the selection order.
 
            // Cal neural network output
            for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
                float in=0;
                for(j=0;j<NumIPs;j++){
                    //           cout<< j << "," <<  i << w1[j][i] << " ";
                    in+=w1[j][i]*x[p][j];
                }
                h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
            }
            
            
            for(i=0; i<NumHN2;i++) { // Cal O/P of hidden layer 2
                float in=0;
                for(j=0;j<NumHN1;j++){
                    in+=w_h1[j][i]*h1[j];
                }
                h2[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
            }
            
            for(i=0; i<NumHN3;i++) { // Cal O/P of hidden layer 3
                float in=0;
                for(j=0;j<NumHN2;j++){
                    in+=w_h2[j][i]*h2[j];
                }
                h3[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
            }
            
            for(i=0;i<NumOPs;i++){ // Cal O/P of output layer
                float in=0;
                for(j=0;j<NumHN3;j++){
                    //            cout<< j << "," <<  i << w2[j][i] << " ";
                    in+=w2[j][i]*h3[j];
                }
                y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
            }
            // Cal error for this pattern
            PatErr=0.0;
            for(i=0;i<NumOPs;i++){
                //          cout<<y[i] << ": " << d[p][i] << " ";
                float err=y[i]-d[p][i]; // actual-desired O/P
                if(err>0)PatErr+=err; else PatErr-=err;
                NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
            }
            if(PatErr<MinErr)MinErr=PatErr;
            if(PatErr>MaxErr)MaxErr=PatErr;
            AveErr+=PatErr;
            
            float tmp;
            // Learn pattern with back propagation
            
            for(i=0;i<NumOPs;i++){ // Modify wts from the hidden layer to output layer;
                ad2[i]=(d[p][i]-y[i])*y[i]*(1.0-y[i]);
                for(j=0;j<NumHN3;j++){
                    tmp = w2[j][i];
                    w2[j][i]+=LrnRate*h3[j]*ad2[i] +
                    Mtm1*(w2[j][i]-w22[j][i]) +
                    Mtm2*(w22[j][i]-w222[j][i]);
                    w222[j][i]=w22[j][i];
                    w22[j][i]=tmp;
                }
            }
            
            
            for(i=0;i<NumHN3;i++){ // Modify wts from the hidden layer2 to hidden layer1
                float err=0.0;
                for(j=0;j<NumOPs;j++)   //error
                    err+=ad2[j]*w22[i][j];
                ad_h2[i]=err*h3[i]*(1.0-h3[i]);
                for(j=0;j<NumHN2;j++){
                    tmp = w_h2[j][i];
                    w_h2[j][i]+=LrnRate*h2[j]*ad_h2[i] +
                    Mtm1*(w_h2[j][i]-w_h22[j][i]) +
                    Mtm2*(w_h22[j][i]-w_h222[j][i]);
                    w_h222[j][i]=w_h22[j][i];
                    w_h22[j][i]=tmp;
                }
            }
            
            
            for(i=0;i<NumHN2;i++){ // Modify wts from the hidden layer2 to hidden layer1
                float err=0.0;
                for(j=0;j<NumHN3;j++)   //error
                    err+=ad_h2[j]*w_h22[i][j];
                ad_h1[i]=err*h2[i]*(1.0-h2[i]);
                for(j=0;j<NumHN1;j++){
                    tmp = w_h1[j][i];
                    w_h1[j][i]+=LrnRate*h1[j]*ad_h1[i] +
                    Mtm1*(w_h1[j][i]-w_h11[j][i]) +
                    Mtm2*(w_h11[j][i]-w_h111[j][i]);
                    w_h111[j][i]=w_h11[j][i];
                    w_h11[j][i]=tmp;
                }
            }
            
            for(i=0;i<NumHN1;i++){ // Modify wts from input layer to the hidden layer;
                float err=0.0;
                for(j=0;j<NumHN2;j++)
                    err+=ad_h1[j]*w_h11[i][j];
                ad1[i]=err*h1[i]*(1.0-h1[i]);
                for(j=0;j<NumIPs;j++){
                    tmp = w1[j][i];
                    w1[j][i]+=LrnRate*x[p][j]*ad1[i] +
                    Mtm1*(w1[j][i]-w11[j][i]) +
                    Mtm2*(w11[j][i]-w111[j][i]);
                    w111[j][i]=w11[j][i];
                    w11[j][i]=tmp;
                }
            }
        }// end for each pattern
        ItCnt++;
        AveErr/=NumPats;
        float PcntErr = NumErr/float(NumPats) * 100.0;
//        fin << PcntErr << endl;
        cout.setf(ios::fixed|ios::showpoint);
        cout<<setprecision(6)<<setw(6)<<ItCnt<<": "<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<<PcntErr<<endl;
        
        if((AveErr<=ObjErr)||(ItCnt==NumIts)) break;
    }// end main learning loop
    // Free memory
//    fin.close();
    delete[] h1; delete[] y;
    delete[] ad1; delete[] ad2;
    delete[] indexArray;
}


// TestNet layer 3
void TestNet3(float **x,float **d,int NumIPs,int NumOPs,int NumPats ){
    
    float *h1 = new float[NumHN1]; // O/Ps of hidden layer
    float *y  = new float[NumOPs]; // O/P of Net
    float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
    int p,i,j;     // for loops indexes
    long ItCnt=0;  // Iteration counter
    long NumErr=0; // Error counter (added for spiral problem)
    
    MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;
    for(p=0;p<NumPats;p++){ // for each pattern... update every step
        // Cal neural network output
        for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
            float in=0;
            for(j=0;j<NumIPs;j++){
                //           cout<< j << "," <<  i << w1[j][i] << " ";
                in+=w1[j][i]*x[p][j];
            }
            h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        }
        //        cout<< endl;
        for(i=0;i<NumOPs;i++){ // Cal O/P of output layer
            float in=0;
            for(j=0;j<NumHN1;j++){
                //            cout<< j << "," <<  i << w2[j][i] << " ";
                in+=w2[j][i]*h1[j];
            }
            y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        }
        // Cal error for this pattern
        PatErr=0.0;
        for(i=0;i<NumOPs;i++){
            //          cout<<y[i] << ": " << d[p][i] << " ";
            float err=y[i]-d[p][i]; // actual-desired O/P
            if(err>0)PatErr+=err; else PatErr-=err;
            NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
        }
        if(PatErr<MinErr)MinErr=PatErr;
        if(PatErr>MaxErr)MaxErr=PatErr;
        AveErr+=PatErr;
        
    }// end for each pattern
    ItCnt++;
    AveErr/=NumPats;
    float PcntErr = NumErr/float(NumPats) * 100.0;
    cout.setf(ios::fixed|ios::showpoint);
    cout<<"test training" <<endl;
    cout<<setprecision(6)<<setw(6)<<"   "<<setw(12)<<"MinErr"<<setw(12)<<"AveErr"<<setw(12)<<"MaxErr"<<setw(12)<<"%error"<<endl;
    cout<<setprecision(6)<<setw(6)<<ItCnt<<": "<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<<PcntErr<<endl;
    
    
    
    
}


// TestNet layer 4
void TestNet4(float **x,float **d,int NumIPs,int NumOPs,int NumPats ){
    
    float *h1 = new float[NumHN1]; // O/Ps of hidden layer
    float *h2 = new float[NumHN2]; // O/Ps of hidden layer2
    float *y  = new float[NumOPs]; // O/P of Net
    float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
    int p,i,j;     // for loops indexes
    long NumErr=0; // Error counter (added for spiral problem)
    
    MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;
    for(p=0;p<NumPats;p++){ // for each pattern... update every step
       
        for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
            float in=0;
            for(j=0;j<NumIPs;j++){
                //           cout<< j << "," <<  i << w1[j][i] << " ";
                in+=w1[j][i]*x[p][j];
            }
            h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        }
        
        
        for(i=0; i<NumHN2;i++) { // Cal O/P of hidden layer 2
            float in=0;
            for(j=0;j<NumHN1;j++){
                in+=w_h1[j][i]*h1[j];
            }
            h2[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        }
        
        
        for(i=0;i<NumOPs;i++){ // Cal O/P of output layer
            float in=0;
            for(j=0;j<NumHN2;j++){
                //            cout<< j << "," <<  i << w2[j][i] << " ";
                in+=w2[j][i]*h2[j];
            }
            y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        }
        // Cal error for this pattern
        PatErr=0.0;
//        cout << y[i] << " : " << d[p][i] << " ; ";
        for(i=0;i<NumOPs;i++){
//            cout<< d[p][i] << " : " << y[i] << " ; " ;
            float err=y[i]-d[p][i]; // actual-desired O/P
            if(err>0)PatErr+=err; else PatErr-=err;
            NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
        }
        if(PatErr<MinErr)MinErr=PatErr;
        if(PatErr>MaxErr)MaxErr=PatErr;
        AveErr+=PatErr;
        
    }// end for each pattern
    AveErr/=NumPats;
    float PcntErr = NumErr/float(NumPats) * 100.0;
    cout.setf(ios::fixed|ios::showpoint);
    cout<<"test training" <<endl;
    cout<<setprecision(6)<<setw(6)<<"   "<<setw(12)<<"MinErr"<<setw(12)<<"AveErr"<<setw(12)<<"MaxErr"<<setw(12)<<"%error"<<endl;
    cout<<setprecision(6)<<setw(6)<<"  "<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<<PcntErr<<endl;
    
    
}


// TestNet layer 5
void TestNet5(float **x,float **d,int NumIPs,int NumOPs,int NumPats ){
    
    float *h1 = new float[NumHN1]; // O/Ps of hidden layer
    float *h2 = new float[NumHN2]; // O/Ps of hidden layer2
    float *h3 = new float[NumHN3]; // O/Ps of hidden layer2
    float *y  = new float[NumOPs]; // O/P of Net
    float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
    int p,i,j;     // for loops indexes
    long NumErr=0; // Error counter (added for spiral problem)
    
    MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;
    for(p=0;p<NumPats;p++){ // for each pattern... update every step
        
        // Cal neural network output
        for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
            float in=0;
            for(j=0;j<NumIPs;j++){
                //           cout<< j << "," <<  i << w1[j][i] << " ";
                in+=w1[j][i]*x[p][j];
            }
            h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        }
        
        
        for(i=0; i<NumHN2;i++) { // Cal O/P of hidden layer 2
            float in=0;
            for(j=0;j<NumHN1;j++){
                in+=w_h1[j][i]*h1[j];
            }
            h2[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        }
        
        for(i=0; i<NumHN3;i++) { // Cal O/P of hidden layer 3
            float in=0;
            for(j=0;j<NumHN2;j++){
                in+=w_h2[j][i]*h2[j];
            }
            h3[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        }
        
        for(i=0;i<NumOPs;i++){ // Cal O/P of output layer
            float in=0;
            for(j=0;j<NumHN3;j++){
                //            cout<< j << "," <<  i << w2[j][i] << " ";
                in+=w2[j][i]*h3[j];
            }
            y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
        }
        // Cal error for this pattern
        PatErr=0.0;
        for(i=0;i<NumOPs;i++){
            //          cout<<y[i] << ": " << d[p][i] << " ";
            float err=y[i]-d[p][i]; // actual-desired O/P
            if(err>0)PatErr+=err; else PatErr-=err;
            NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
        }
        if(PatErr<MinErr)MinErr=PatErr;
        if(PatErr>MaxErr)MaxErr=PatErr;
        AveErr+=PatErr;
        
    }// end for each pattern
    AveErr/=NumPats;
    float PcntErr = NumErr/float(NumPats) * 100.0;
    cout.setf(ios::fixed|ios::showpoint);
    cout<<"test training" <<endl;
    cout<<setprecision(6)<<setw(6)<<"   "<<setw(12)<<"MinErr"<<setw(12)<<"AveErr"<<setw(12)<<"MaxErr"<<setw(12)<<"%error"<<endl;
    cout<<setprecision(6)<<setw(6)<<"  "<<setw(12)<<MinErr<<setw(12)<<AveErr<<setw(12)<<MaxErr<<setw(12)<<PcntErr<<endl;
    
    
}


/**
 画出双螺旋选的分布图 两层hidden layer
 */
void DrawGraph(int NumIPs,int NumOPs){
    
    float *h1 = new float[NumHN1]; // O/Ps of hidden layer
    float *h2 = new float[NumHN2]; // O/Ps of hidden layer2
    float *y  = new float[NumOPs]; // O/P of Net
    float *x  = new float[NumIPs];
    int i,j;     // for loops indexes
    
    
    float min_x = -1; float max_x = 1; float gap_x = 0.02;
    float min_y = -1; float max_y = 1; float gap_y = 0.02;
    
    for(float pox_y = min_y ; pox_y<max_y ; pox_y+=gap_y){
        
        for(float pox_x = min_x; pox_x<max_x; pox_x+=gap_x){
            
            x[0] = pox_x;
            x[1] = pox_y;
            
                for(i=0;i<NumHN1;i++){ // Cal O/P of hidden layer 1
                    float in=0;
                    for(j=0;j<NumIPs;j++){
                        //           cout<< j << "," <<  i << w1[j][i] << " ";
                        in+=w1[j][i]*x[j];
                    }
                    h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
                }
            
            
                for(i=0; i<NumHN2;i++) { // Cal O/P of hidden layer 2
                    float in=0;
                    for(j=0;j<NumHN1;j++){
                        in+=w_h1[j][i]*h1[j];
                    }
                    h2[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
                }
            
            
                for(i=0;i<NumOPs;i++){ // Cal O/P of output layer
                    float in=0;
                    for(j=0;j<NumHN2;j++){
//                        cout<< j << "," <<  i << w2[j][i] << " ";
                        in+=w2[j][i]*h2[j];
                    }
                    y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
                }
            
            if (y[0] < 0.5) {cout<<"-";} else{cout << "X";}
            
            
        }
        
        cout << endl;
    }
    
    
    
    
    
    
}


float **Aloc2DAry(int m,int n){
//Allocates memory for 2D array
  float **Ary2D = new float*[m];
  if(Ary2D==NULL){cout<<"No memory!\n";exit(1);}
  for(int i=0;i<m;i++){
	 Ary2D[i] = new float[n];
	 if(Ary2D[i]==NULL){cout<<"No memory!\n";exit(1);}
  }
  return Ary2D;
}

void Free2DAry(float **Ary2D,int n){
//Frees memory in 2D array
  for(int i=0;i<n;i++)
	 delete [] Ary2D[i];
  delete [] Ary2D;
}


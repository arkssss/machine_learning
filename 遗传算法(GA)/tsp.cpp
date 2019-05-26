/************************************************************
 *      tsp.cpp - GA Program for CSCI964 - Ass3
 *      Create by: <Zhou Fang & 6286914>
 *************************************************************/
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <map>
#include <cmath>
using namespace std;
// def class of city
class City{
    public :
        int x_pos;
        int y_pos;
        int type;
    public :
    // construstor
    City(){}
    // construstor
    City(int x_pos, int y_pos, int type):x_pos(x_pos), y_pos(y_pos), type(type){}
    
    // output reload
    friend ostream& operator<<(ostream& out,const City& c){
        out << "x_pos: " << c.x_pos << " y_pos: " << c.y_pos << " type: "<< c.type<<endl;
        return out;
    }
    
    bool operator==(const City& c){
        return (x_pos==c.x_pos) && (y_pos==c.y_pos) && (type==c.type);
    }
    
};

// global var
enum MutationType{OnePoint, TwoPoint, ThreePoint, ReducePoint};  // the type of mutation    // ReducePoint means the mutation reduce after sevral Gen
enum SelectType{tournament, roulettewheel};          // the type of Selection Winner
vector<City> AllCity;                                 // store all the city
int cIndividualLength;                               // length of DNA which is the number of city
const int cPopSize = 150;                            // must be a even number cause the cross over
const int cGen = 2000;                                // how many round should be taken
const int Seed = 321;                                // seed for random number
const int cTournamentSize = 7;                       // the round of tournament
const double cCrossoverRate = 0.75;
const double cMutationRate = 0.1;
const SelectType SType = tournament;        // define the type of selection
const MutationType Mtype = ReducePoint;        // define the type of mutation
const double cCrossOverLengthRate = 0.75;   // the length need to be cross
map<pair<int, int>, int> type_cost;        // store the mapping between the different type with the diff cost
map<pair<int, int>, double> look_up;         // cache to store the comptued distance
const bool IS_WRITE = false;                  // if record the res to a file
const bool IS_REDUCE_CROSSOVER = true;       // true means every gen reduce the CrossOver length

// functions
void read_city(char* file_url);     // read the data and store it into the vector
void write_res(vector<int> BestMember,char *member_file, double BestFitness, char *Fitness_file); // write to file
void init_cost_map();               // init the type_cost map
void init_pop(vector<vector<int> > &CurrentPop, vector<vector<int> > &NextPop, vector<double> &Fitness, vector<int> &BestMember);   // init two pop and Finess and BestMember
double EvaluateFitness(vector<int> Member);
double get_distacne(City c1, City c2, int index1, int index2);
int Selection(vector<double> Fitness);
int Tournament(vector<double> Fitness);
int RouletteWheel(vector<double> Fitness);
int CurtCrossOverLength(int CurtGen);        // compute the CurtCrosssOverLength Make it smaller every Gen
void CrossOver(vector<int> parent1, vector<int> parent2, vector<int> &child1, vector<int> &child2, int Gen);
void Mutate(vector<int> &member, int CurtGen);
bool IsExistElem(vector<int> v, int i);     // i if exist in v
int  RandInt(int n);
double Rand01();    // 0..1


int main(){
    
    char file_url[] ="dataset/tsp500.txt";
    char member_file[] = "res/tsp100_bestMember.txt";
    char bestFitness_file[] = "res/tsp100_bestFitness.txt";
    
    
    read_city(file_url);

    
    init_cost_map();
    
    // init the pop
    vector<vector<int> > CurentPop;
    vector<vector<int> > NextPop;
    vector<double> Fitness;             // the des all the DNA's Fitness number
    vector<int> BestMember;             // BestMember des the best one with Fitness
    
    // init the pop
    init_pop(CurentPop, NextPop, Fitness, BestMember);
    
    double BestFitness = __DBL_MAX__;
    
    // main loop
    for(int CurGen=0; CurGen<cGen; CurGen++){

        for(int j=0; j<cPopSize; j++){
            
            // Evaluate the fitness
            Fitness[j] = EvaluateFitness(CurentPop[j]);
            // find best
            if(Fitness[j] < BestFitness){
                BestFitness = Fitness[j];
                BestMember  = CurentPop[j];
            }
        
        }
        
        // Next Gen
        for(int i=0; i<cPopSize; i+=2){
            int parent1 = Selection(Fitness);
            int parent2 = Selection(Fitness);
            if(Rand01() < cCrossoverRate) {
                // cCrossover
                CrossOver(CurentPop[parent1], CurentPop[parent2], NextPop[i], NextPop[i+1], CurGen);
            }else{
                // the same
                NextPop[i] = CurentPop[parent1];
                NextPop[i+1] = CurentPop[parent2];
            }
            if(Rand01() < cMutationRate){
                // mutation i
                Mutate(NextPop[i], CurGen);
            }
            if(Rand01() < cMutationRate){
                // mutation i+1
                Mutate(NextPop[i+1], CurGen);
            }
        }
        
        cout<<"Gen: " << CurGen << " BestFintness: " <<BestFitness <<endl;
        if(IS_WRITE) write_res(BestMember, member_file, BestFitness, bestFitness_file);
        
        //evolution
        CurentPop = NextPop;
        for(int i=0; i<cPopSize; i++) NextPop[i].clear();
    }
}

void read_city(char* file_url){
    
    ifstream fin;
    int city_number;
    int x_pos;
    int y_pos;
    int type;
    
    fin.open(file_url);
    if(!fin.good()) {cout<< "File not exist!" << endl;exit(1);}
    
    char Line[500];
    
    // fisrt line
    fin.getline(Line, 500);
    sscanf(Line,"%d", &city_number);
    while(fin.getline(Line, 500)){
        sscanf(Line, "%d %d %d", &x_pos, &y_pos, &type);
        
        // store
        AllCity.push_back(City(x_pos, y_pos, type));
    }
    
    if (city_number != AllCity.size()) {cout<<"something wrong with the read process!"; exit(1);}
    cIndividualLength = city_number;
    fin.close();
    return ;
}

void write_res(vector<int> BestMember, char *member_file, double BestFitness, char *Fitness_file){
    ofstream fout_member;
    ofstream fout_fitness;
    
    fout_member.open(member_file, ios::out | ios::app);
    fout_fitness.open(Fitness_file, ios::out | ios::app);
    
    if(!fout_fitness.good() || !fout_member.good()){
        cout << "something wrong with the out file";
        exit(1);
    }
    
    string Member = "";
    for(int i=0; i<BestMember.size() ;i++){
        Member += to_string(BestMember[i]) + " ";
    }
    
    fout_member << Member << endl;
    fout_fitness << BestFitness << endl;
    
    fout_member.close();
    fout_fitness.close();
    
    return;
}

void init_cost_map(){
    type_cost[pair<int ,int>(1, 1)] = 10;
    type_cost[pair<int ,int>(1, 2)] = type_cost[pair<int ,int>(2, 1)] = 7.5;
    type_cost[pair<int ,int>(2, 2)] = 5;
    type_cost[pair<int ,int>(3, 1)] = type_cost[pair<int ,int>(1, 3)] = 5;
    type_cost[pair<int ,int>(3, 2)] = type_cost[pair<int ,int>(2, 3)] = 2.5;
    type_cost[pair<int ,int>(3, 3)] = 1;
}

void init_pop(vector<vector<int> > &CurrentPop, vector<vector<int> > &NextPop, vector<double> &Fitness, vector<int> &BestMember){
    srand(Seed);
    // init vector with 0
    NextPop.resize(cPopSize);
    Fitness.resize(cPopSize);
    BestMember.resize(cIndividualLength);
    
    vector<int> init_path;
    // init path
    for(int i=0; i<cIndividualLength; i++){init_path.push_back(i);}
    
    for(int i=0; i<cPopSize; i++){
        vector<int> tmp_path = init_path;
        
        // init to the random path
        for(int j=0; j<cIndividualLength; j++){
            int random_pos = RandInt(cIndividualLength);
            // swap
            int tmp = tmp_path[j];
            tmp_path[j] = tmp_path[random_pos];
            tmp_path[random_pos] = tmp;
        }
        CurrentPop.push_back(tmp_path);
    }
    return ;
}

// compute the fitness of one member
double EvaluateFitness(vector<int> Member){
    unsigned long length = Member.size();
    double fitness = 0;
    for(int i=0; i<length-1; i++){
        fitness += get_distacne(AllCity[Member[i]], AllCity[Member[i+1]], Member[i], Member[i+1]);
    }
    
    return fitness;
}

double get_distacne(City c1, City c2, int index1, int index2){
    
    if (look_up.count(pair<int, int>(index1, index2))) return look_up[pair<int, int>(index1, index2)];
    
    double dis;
    dis = sqrt(pow(c1.x_pos - c2.x_pos, 2) + pow(c1.y_pos - c2.y_pos, 2));
    dis *= type_cost[pair<int, int>(c1.type, c2.type)];
    look_up[pair<int, int>(index1, index2)] = look_up[pair<int, int>(index2, index1)] = dis;
    
    return dis;
}

int Selection(vector<double> Fitness){
    switch (SType) {
        case tournament:{
            return Tournament(Fitness);
            break;
        }
        case roulettewheel:{
            return  RouletteWheel(Fitness);
            break;
        }
        default:
            cout << "Wrong Selection type!";
            exit(1);
            break;
    }
}

// get the index if the winner DNA
int Tournament(vector<double> Fitness){
    double WinFit = __DBL_MAX__;
    int Winner;
    for(int i=0; i<cTournamentSize; i++){
        int random_member = RandInt(cPopSize);
        if(Fitness[random_member] < WinFit){
            WinFit = Fitness[random_member];
            Winner = random_member;
        }
    }
    return Winner;
}

int RouletteWheel(vector<double> Fitness){
    
    // find the min and max number in Fitness
    double min_f = Fitness[0], max_f = Fitness[0];
    // index
    int max_i = 0;
    
    for(int i=1; i<cPopSize; i++){
        if(Fitness[i] > max_f){
            max_f = Fitness[i];
            max_i = i;
            continue;
        }
        if(Fitness[i] < min_f){
            min_f = Fitness[i];
            min_f = i;
        }
    }
    // normalised
    vector<double> Norm_Fitness(cPopSize);
    for(int i=0; i<cPopSize; i++){
        Norm_Fitness[i] = Fitness[i] - min_f;
    }
    
    // Reverse Norm_Fitness
    // This is important cause in this experments The Lower Fitness mean the better answer
    // Thus we need to put the lower Fitness to have higher possbility to be choosen
    double norm_max = Norm_Fitness[max_i];
    // liner substart
    double Tot = 0;
    for(int i=0; i<cPopSize; i++){
        // make lower fitness high possbility
        Norm_Fitness[i] = norm_max - Norm_Fitness[i];
        Tot += Norm_Fitness[i];
    }
    // make it to possbility [0,1]
    for(int i=0; i<cPopSize; i++){Norm_Fitness[i] /= Tot;}
    
    // now we get the Roulette Wheel
    double possibility = Rand01();
    double m = 0;
    for(int i=0; i<cPopSize; i++){
        m += Norm_Fitness[i];
        if (possibility < m)
            // get winner
        {
//            cout << i << endl;
            return i;
        }
    }
    cout << "Something Wrong in the Roulette Wheel";
    exit(1);
}

int CurtCrossOverLength(int CurtGen){
    
    
    double CurtCrossOverLengthRate = IS_REDUCE_CROSSOVER ? ((cGen - CurtGen)/cGen) * cCrossOverLengthRate : cCrossOverLengthRate;
    
    return cIndividualLength * CurtCrossOverLengthRate;
    
}

void CrossOver(vector<int> parent1, vector<int> parent2, vector<int> &child1, vector<int> &child2, int Gen){
    
    // cCrossOverLengthRate need to be smaller and smaller
    int CrossOverLength = CurtCrossOverLength(Gen);
    
    // random to cross over
    // select the points which need to be swapped
    vector<int> CrossChoose;
    for (int i=0; i<cIndividualLength; i++) CrossChoose.push_back(i);
    // random swap it
    for (int i=0; i<CrossChoose.size();i++){
        int ran_pos = RandInt(cIndividualLength);
        int tmp = CrossChoose[i];
        CrossChoose[i] = CrossChoose[ran_pos];
        CrossChoose[ran_pos]= tmp;
    }
    
    // the front CrossOverLength is the swap points' pos
    vector<int> CrossPoints;
    for(int i=0; i<CrossOverLength; i++){CrossPoints.push_back(CrossChoose[i]);}
    
    // start cross over
    child1.clear();
    child2.clear();
    
    // Cross Over
    for(int i=0;i<CrossOverLength;i++){
        child1.push_back(parent1[CrossPoints[i]]);
        child2.push_back(parent2[CrossPoints[i]]);
    }
    for(int i=0;i<cIndividualLength;i++){
        if(!IsExistElem(child1, parent2[i])){
            child1.push_back(parent2[i]);
        }
        if(!IsExistElem(child2, parent1[i])){
            child2.push_back(parent1[i]);
        }
    }
    
    
    if (child1.size()!=cIndividualLength || child2.size()!=cIndividualLength){
        cout<< "Something Wrong When Crossing Over";
        exit(1);
    }
    
    return ;
    
    
}

bool IsExistElem(vector<int> v, int i){
    
    vector<int>::iterator it;
    it = std::find(v.begin(), v.end(), i);
    if(it == v.end()){
        return false;
    }else{
        return true;
    }
}

// Mutation
void Mutate(vector<int> &member, int CurtGen){
    
    MutationType CurtMtype = Mtype;
    
    if (CurtMtype == ReducePoint){
        
        int Threshold = cGen * 0.333;
        
        switch (CurtGen / Threshold) {
            case 0:{
                CurtMtype = ThreePoint;
                break;
            }
            case 1:{
                CurtMtype = TwoPoint;
                break;
            }
            case 2:{
                CurtMtype = OnePoint;
                break;
            }
            default:
                CurtMtype = OnePoint;
        }
    };
    
    switch(CurtMtype) {
        case OnePoint: {
            int random_pos1, random_pos2;
            random_pos1 = RandInt(cIndividualLength);
            // random_pos2 should different from random_pos1
            do{random_pos2 = RandInt(cIndividualLength);}while(random_pos2 == random_pos1);
            
            // start mutation
            int tmp = member[random_pos1];
            member[random_pos1] = member[random_pos2];
            member[random_pos2] = tmp;
            break;}
        case TwoPoint:{
            int r_pos1, r_pos2, r_pos3, r_pos4;
            r_pos1 = RandInt(cIndividualLength);
            do{r_pos2 = RandInt(cIndividualLength);}while(r_pos2 == r_pos1);
            do{r_pos3 = RandInt(cIndividualLength);}while(r_pos3 == r_pos2 || r_pos3 == r_pos1);
            do{r_pos4 = RandInt(cIndividualLength);}while(r_pos4 == r_pos3 || r_pos4 == r_pos2 || r_pos4 == r_pos1);
            
            // start mutation
            // r_pos1 <=> r_pos2
            int tmp = member[r_pos1];
            member[r_pos1] = member[r_pos2];
            member[r_pos2] = tmp;
            
            // r_pos3 <=> rr_pos4
            tmp = member[r_pos3];
            member[r_pos3] = member[r_pos4];
            member[r_pos4] = tmp;
            break;
            }
        case ThreePoint:{
            int r_pos1, r_pos2, r_pos3, r_pos4, r_pos5, r_pos6;
            r_pos1 = RandInt(cIndividualLength);
            do{r_pos2 = RandInt(cIndividualLength);}while(r_pos2 == r_pos1);
            do{r_pos3 = RandInt(cIndividualLength);}while(r_pos3 == r_pos2 || r_pos3 == r_pos1);
            do{r_pos4 = RandInt(cIndividualLength);}while(r_pos4 == r_pos3 || r_pos4 == r_pos2 || r_pos4 == r_pos1);
            do{r_pos5 = RandInt(cIndividualLength);}while(r_pos5 == r_pos4 || r_pos5 == r_pos3 || r_pos5 == r_pos2 || r_pos5 == r_pos1);
            do{r_pos6 = RandInt(cIndividualLength);}while(r_pos6 == r_pos5 || r_pos6 == r_pos4 || r_pos6 == r_pos3 || r_pos6 == r_pos2 || r_pos6 == r_pos1);
            // start mutation
            // r_pos1 <=> r_pos2
            int tmp = member[r_pos1];
            member[r_pos1] = member[r_pos2];
            member[r_pos2] = tmp;
            
            // r_pos3 <=> rr_pos4
            tmp = member[r_pos3];
            member[r_pos3] = member[r_pos4];
            member[r_pos4] = tmp;
            
            // r_pos5 <=> r_pos6
            tmp = member[r_pos5];
            member[r_pos5] = member[r_pos6];
            member[r_pos6] = tmp;
            break;
            }
        default:{
            cout<<"Unknown Mutation Type!";
            exit(1);
            break;
        }
    }
    
    return ;
}

int RandInt(int n){ // 0..n-1
    return int( rand()/(double(RAND_MAX)+1) * n );
}

double Rand01(){ // 0..1
    return(rand()/(double)(RAND_MAX));
}

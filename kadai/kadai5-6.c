#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int getYourHand(){
  int a;
  while(1){
    printf("Your input (0,2,5): ");
    int t = scanf("%d",&a);
    if(a==0 || a==2 || a==5){
      break;
    }else{
      printf("Invalid input => Input again.\n");
      if(!t){
        scanf("%*s");
      }
    }
  }
  return a;
}

int randComHand(){
  int b = rand()%3;
  b = b*2 + (b == 2);
  return b;
}

int main(){
  srand(time(NULL));
  int you,com,rslt;
  char judge[3][30] = {"Try again.","You lose.","You win."};
  do{
    you = getYourHand();
    com = randComHand();
    //result
    rslt = (((you/2 + 1)%3) == com/2) + (com != you);
    //print
    printf("Comp:%d vs You:%d => %s\n",com,you,judge[rslt]);
  }while(!rslt);
  return 0;
}

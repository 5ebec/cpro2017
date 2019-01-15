#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  int a,b,c,you,comp;
  char judge[3][30] = {"Try again.","You lose.","You win."};
  first:
  srand(time(NULL));
  b=rand()%3;
  printf("Your input (0,2,5): ");
  scanf("%d",&you);
  //you: 0->0,2->1,5->2
  if(you==0||you==2||you==5){
    a = you/2;
  }else{
    printf("Invalid input => Input again.\n");
    goto first;
  }
  //result
  c = ((((a+1)%3)==b)+(b!=a));
  //computer: 0->0,1->2,2->5
  comp = b*2+(b==2);
  //print
  printf("Comp:%d vs You:%d => %s\n",comp,you,judge[c]);
  //try again
  if(c == 0){
    goto first;
  }
  return 0;
}

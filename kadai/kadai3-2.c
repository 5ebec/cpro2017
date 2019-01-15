#include <stdio.h>

int main(){
  int i=1,n,ans=1;
  printf("enter a number: ");
  scanf("%d",&n);
  while(i<=n){
    ans = ans * i;
    i++;
  }
  printf("%d!=%d\n",n,ans);
  return 0;
}

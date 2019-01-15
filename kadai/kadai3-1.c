#include <stdio.h>

int main(){
  int n,ans=1;
  printf("enter a number: ");
  scanf("%d",&n);
  for(int i=1; i<=n; i++){
    ans = ans * i;
  }
  printf("%d!=%d\n",n,ans);
  return 0;
}

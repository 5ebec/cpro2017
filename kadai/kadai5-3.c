#include <stdio.h>

int main(){
  char c = 'a';
  printf("%c %d %x\n", c,c,c);
  c++;
  printf("%c %d %x\n", c,c,c);
  return 0;
}

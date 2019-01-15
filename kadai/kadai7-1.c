#include <stdio.h>

int main() {
  char str[128], *i;
  printf("Input a word: ");
  scanf("%s", str);
  for(i = str; *i; i++){
    if((*i >= 'a' && *i <= 'y')||(*i >= 'A' && *i <= 'Y')){
      *i += 1;
    }else if(*i == 'z' || *i == 'Z'){
      *i -= 25;
    }
  }
  printf("%s\n", str);
  return 0;
}

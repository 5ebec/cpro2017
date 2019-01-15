#include <stdio.h>

int main(){
  FILE *fp_in,*fp_out;
  char str[128];
  fp_in = fopen("test.txt","r");
  if(!fp_in){
    printf("File cannot open\n");
    return 0;
  }else{
    fp_out = fopen("test.bak","w");
    while(fgets(str,128,fp_in)){
      fprintf(fp_out,"%s",str);
    }
    fclose(fp_in);
    fclose(fp_out);
  }
  return 0;
}

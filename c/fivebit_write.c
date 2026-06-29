/* fivebit_write.c — group-commit durable write in C.
 * Encode N two-int records (C codec), append packed bytes, fsync once per batch B.
 * Build: gcc -O2 -o fivebit_write fivebit_write.c
 * Run:   fivebit_write <path> <N> <B>   -> prints  rate  fsyncs
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

enum { T_D0=0, T_N1=17, T_RECORD=28, T_END=30 };

static int pack(const uint8_t*tk,int n,uint8_t*out){
    uint32_t acc=0; int nbits=0,len=0;
    for(int i=0;i<n;i++){ acc=(acc<<5)|(tk[i]&0x1F); nbits+=5; while(nbits>=8){ nbits-=8; out[len++]=(uint8_t)((acc>>nbits)&0xFF);} }
    if(nbits>0) out[len++]=(uint8_t)((acc<<(8-nbits))&0xFF);
    return len;
}
static void enc_int(long long v,uint8_t*tk,int*n){
    if(v==0){ tk[(*n)++]=T_D0; tk[(*n)++]=T_END; return; }
    int neg=v<0; unsigned long long a=neg?(unsigned long long)(-(v+1))+1ULL:(unsigned long long)v;
    char d[32]; int k=0; while(a){ d[k++]=(char)('0'+a%10); a/=10; }
    for(int i=k-1;i>=0;i--){ int dig=d[i]-'0'; tk[(*n)++]= dig==0?T_D0:(uint8_t)(neg?16+dig:dig);} tk[(*n)++]=T_END;
}
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

int main(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: %s path N B\n",argv[0]); return 1; }
    const char*path=argv[1]; long N=atol(argv[2]); int B=atoi(argv[3]);
    int fd=open(path,O_WRONLY|O_CREAT|O_TRUNC,0644);
    if(fd<0){ perror("open"); return 1; }

    uint8_t wbuf[1<<20]; int wlen=0; long fsyncs=0; int in_batch=0;
    double t0=now();
    for(long i=0;i<N;i++){
        uint8_t tk[64]; int n=0; enc_int(i,tk,&n); enc_int(i*7-3,tk,&n); tk[n++]=T_RECORD;
        uint8_t rec[64]; int rl=pack(tk,n,rec);
        if(wlen+rl>(int)sizeof(wbuf)){ write(fd,wbuf,wlen); wlen=0; }
        memcpy(wbuf+wlen,rec,rl); wlen+=rl;
        if(++in_batch>=B){ if(wlen){ write(fd,wbuf,wlen); wlen=0; } fsync(fd); fsyncs++; in_batch=0; }
    }
    if(wlen) write(fd,wbuf,wlen);
    fsync(fd); fsyncs++;
    close(fd);
    double dt=now()-t0;
    printf("%.0f\t%ld\n", N/dt, fsyncs);
    return 0;
}

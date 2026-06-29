/* fivebit_codec.c — encode AND decode, ported to match python/binary_grid_db.py.
 * Build: gcc -O2 -o fivebit_codec fivebit_codec.c
 * Modes:
 *   conf            print hex+pad for the encode battery (encoder conformance)
 *   decode          read "hex pad" lines on stdin, print reconstructed items
 *   benc N          pure encode+pack throughput (no fsync)
 *   bdec N          pure unpack+parse throughput (no fsync)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>

enum { T_D0=0, T_PLUS=10, T_N1=17, T_POW=26, T_SCALE=27,
       T_RECORD=28, T_CHECKSUM=29, T_END=30, T_START=31 };

typedef struct { uint8_t *t; int n, cap; } Toks;
static void tpush(Toks *b, uint8_t v){
    if (b->n>=b->cap){ b->cap=b->cap?b->cap*2:64; b->t=realloc(b->t,b->cap);} b->t[b->n++]=v; }

/* ---------------- ENCODE ---------------- */
static void encode_integer(long long value, Toks *out){
    if (value==0){ tpush(out,T_D0); tpush(out,T_END); return; }
    int neg=value<0; unsigned long long a=neg?(unsigned long long)(-(value+1))+1ULL:(unsigned long long)value;
    char d[32]; int n=0; while(a){ d[n++]=(char)('0'+a%10); a/=10; }
    for(int i=n-1;i>=0;i--){ int dig=d[i]-'0'; tpush(out, dig==0?T_D0:(uint8_t)(neg?16+dig:dig)); }
    tpush(out,T_END);
}
static int word_tok(char c){ if(c>='A'&&c<='Z')return c-'A'; if(c==' ')return T_POW; if(c=='.')return T_SCALE; return -1; }
static int special_tok(char c){ if(c>='a'&&c<='z')return c-'a'; if(c=='@')return T_POW; if(c=='-')return T_SCALE; return -1; }
static int special2_tok(char c){ static const char*S="!\"#$%&'()*+,/:;<=>?[\\]^_`{|}"; const char*p=strchr(S,c); return p?(int)(p-S):-1; }
static int encode_word(const char*text, Toks*out){
    tpush(out,T_START); int depth=0;
    for(const char*p=text;*p;p++){ char ch=*p; int v;
        if(isdigit((unsigned char)ch)){ for(int k=0;k<depth;k++)tpush(out,T_END); depth=0; tpush(out,T_END); tpush(out,(uint8_t)(ch-'0')); tpush(out,T_START); continue; }
        if((v=word_tok(ch))>=0){ for(int k=0;k<depth;k++)tpush(out,T_END); depth=0; tpush(out,(uint8_t)v); continue; }
        if((v=special_tok(ch))>=0){ if(depth>1){tpush(out,T_END);depth=1;} else if(depth<1){tpush(out,T_START);depth=1;} tpush(out,(uint8_t)v); continue; }
        if((v=special2_tok(ch))>=0){ if(depth<2){ if(depth<1){tpush(out,T_START);depth=1;} tpush(out,T_START);depth=2;} tpush(out,(uint8_t)v); continue; }
        if(isalpha((unsigned char)ch)){ char u=(char)toupper((unsigned char)ch); for(int k=0;k<depth;k++)tpush(out,T_END); depth=0; tpush(out,(uint8_t)word_tok(u)); continue; }
        return -1;
    }
    for(int k=0;k<depth;k++)tpush(out,T_END); tpush(out,T_END); return 0;
}
static int pack(const uint8_t*tk,int n,uint8_t*out,int*pad_out){
    uint32_t acc=0; int nbits=0,len=0;
    for(int i=0;i<n;i++){ acc=(acc<<5)|(tk[i]&0x1F); nbits+=5; while(nbits>=8){ nbits-=8; out[len++]=(uint8_t)((acc>>nbits)&0xFF);} }
    int pad=0; if(nbits>0){ out[len++]=(uint8_t)((acc<<(8-nbits))&0xFF); pad=8-nbits; }
    if(pad_out)*pad_out=pad; return len;
}

/* ---------------- DECODE ---------------- */
static int unpack(const uint8_t*data,int nbytes,int pad,uint8_t*tok){
    int total_bits=nbytes*8-pad; int ntok=total_bits/5;
    uint32_t acc=0; int nbits=0,produced=0;
    for(int i=0;i<nbytes && produced<ntok;i++){
        acc=(acc<<8)|data[i]; nbits+=8;
        while(nbits>=5 && produced<ntok){ nbits-=5; tok[produced++]=(acc>>nbits)&0x1F; }
    }
    return ntok;
}

/* decoded item sink */
typedef struct { int is_word; long long ival; char word[256]; } Item;

/* char maps for finalize */
static char word_char(int t){ if(t>=0&&t<=25)return 'A'+t; if(t==26)return ' '; if(t==27)return '.'; return '?'; }
static char special_char(int t){ if(t>=0&&t<=25)return 'a'+t; if(t==26)return '@'; if(t==27)return '-'; return '?'; }
static char special2_char(int t){ static const char*S="!\"#$%&'()*+,/:;<=>?[\\]^_`{|}"; return (t>=0&&t<28)?S[t]:'?'; }

/* parse tokens -> items. Returns count. (NUM/WORD/SPECIAL/SPECIAL2; integers+words). */
static int parse(const uint8_t*tk,int n,Item*items,int max_items){
    int st=0; /* 0=NUM 1=WORD 2=SPECIAL 3=SPECIAL2 */
    long long acc_digits[64]; int nd=0;          /* signed digits */
    char wbuf[256]; int wl=0;                     /* word chars */
    int ni=0;
    #define FIN_NUM() do{ if(nd){ long long v=0; for(int k=0;k<nd;k++) v=v*10+acc_digits[k]; if(ni<max_items){items[ni].is_word=0; items[ni].ival=v; ni++;} nd=0; } }while(0)
    #define FIN_WORD(mapfn) do{ if(ni<max_items){ items[ni].is_word=1; for(int k=0;k<wl;k++) items[ni].word[k]=mapfn((unsigned char)wbuf[k]); items[ni].word[wl]=0; ni++; } wl=0; }while(0)
    for(int i=0;i<n;i++){ int t=tk[i];
        if(st==0){
            if(t==T_START){ FIN_NUM(); st=1; }
            else if(t==T_END){ FIN_NUM(); }
            else if(t==T_RECORD){ FIN_NUM(); }
            else if(t<=9){ acc_digits[nd++]=t; }
            else if(t>=T_N1 && t<=25){ acc_digits[nd++]=-(t-16); }
            /* operators/annotations ignored for this codec bench */
        } else if(st==1){
            if(t==T_END){ FIN_WORD(word_char); st=0; }
            else if(t==T_RECORD){ FIN_WORD(word_char); st=0; }
            else if(t==T_START){ FIN_WORD(word_char); st=2; }
            else { wbuf[wl++]=t; }
        } else if(st==2){
            if(t==T_END){ FIN_WORD(special_char); st=1; }
            else if(t==T_RECORD){ FIN_WORD(special_char); st=0; }
            else if(t==T_START){ FIN_WORD(special_char); st=3; }
            else { wbuf[wl++]=t; }
        } else { /* SPECIAL2 */
            if(t==T_END){ FIN_WORD(special2_char); st=2; }
            else if(t==T_RECORD){ FIN_WORD(special2_char); st=0; }
            else { wbuf[wl++]=t; }
        }
    }
    FIN_NUM();
    return ni;
}

/* ---------------- helpers ---------------- */
static int from_hex(const char*hex,uint8_t*out){ int n=strlen(hex)/2; for(int i=0;i<n;i++){ unsigned v; sscanf(hex+2*i,"%2x",&v); out[i]=(uint8_t)v; } return n; }
static void to_hex(const uint8_t*b,int n,char*o){ static const char*H="0123456789abcdef"; for(int i=0;i<n;i++){o[2*i]=H[b[i]>>4];o[2*i+1]=H[b[i]&0xF];} o[2*n]=0; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

static void conformance(void){
    long long ints[]={0,1,-1,42,-42,123,-105,999999,-1000000,7,-7};
    for(size_t i=0;i<sizeof(ints)/sizeof(ints[0]);i++){ Toks tk={0}; encode_integer(ints[i],&tk); uint8_t b[256]; int pad; int len=pack(tk.t,tk.n,b,&pad); char h[600]; to_hex(b,len,h); printf("INT\t%lld\t%s\t%d\n",ints[i],h,pad); free(tk.t);}
    const char*ws[]={"HI","hi","Hi","a!b","a@b","a.b","user 42","id007","Alice","NewYork","A1B2"};
    for(size_t i=0;i<sizeof(ws)/sizeof(ws[0]);i++){ Toks tk={0}; encode_word(ws[i],&tk); uint8_t b[1024]; int pad; int len=pack(tk.t,tk.n,b,&pad); char h[2100]; to_hex(b,len,h); printf("WORD\t%s\t%s\t%d\n",ws[i],h,pad); free(tk.t);}
}

static void decode_stdin(void){
    char line[8192];
    while(fgets(line,sizeof(line),stdin)){
        char hex[8192]; int pad;
        if(sscanf(line,"%s %d",hex,&pad)!=2) continue;
        uint8_t bytes[4096]; int nb=from_hex(hex,bytes);
        uint8_t tok[8192]; int nt=unpack(bytes,nb,pad,tok);
        Item items[256]; int ni=parse(tok,nt,items,256);
        for(int i=0;i<ni;i++){ if(i)printf("|"); if(items[i].is_word)printf("%s",items[i].word); else printf("%lld",items[i].ival); }
        printf("\n");
    }
}

static void bench_encode(int N){
    uint8_t buf[256]; double t0=now(); long long sink=0;
    for(int i=0;i<N;i++){ Toks tk={0}; encode_integer(i,&tk); encode_integer(i*7-3,&tk); tpush(&tk,T_RECORD); int pad; sink+=pack(tk.t,tk.n,buf,&pad); free(tk.t);}
    double dt=now()-t0; printf("C_ENCODE\t%d\t%.6f\t%.0f\n",N,dt,N/dt); (void)sink;
}
static void bench_decode(int N){
    /* pre-encode N records into a contiguous store, then time decode only */
    int pads[200000]; static uint8_t store[200000][24]; int lens[200000];
    if(N>200000)N=200000;
    for(int i=0;i<N;i++){ Toks tk={0}; encode_integer(i,&tk); encode_integer(i*7-3,&tk); tpush(&tk,T_RECORD); lens[i]=pack(tk.t,tk.n,store[i],&pads[i]); free(tk.t);}
    double t0=now(); long long sink=0;
    for(int i=0;i<N;i++){ uint8_t tok[64]; int nt=unpack(store[i],lens[i],pads[i],tok); Item it[8]; int ni=parse(tok,nt,it,8); for(int k=0;k<ni;k++) sink+=it[k].ival; }
    double dt=now()-t0; printf("C_DECODE\t%d\t%.6f\t%.0f\n",N,dt,N/dt); (void)sink;
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: conf|decode|benc N|bdec N\n"); return 1; }
    if(!strcmp(argv[1],"conf")) conformance();
    else if(!strcmp(argv[1],"decode")) decode_stdin();
    else if(!strcmp(argv[1],"benc")) bench_encode(argc>=3?atoi(argv[2]):100000);
    else if(!strcmp(argv[1],"bdec")) bench_decode(argc>=3?atoi(argv[2]):100000);
    else { fprintf(stderr,"unknown mode\n"); return 1; }
    return 0;
}

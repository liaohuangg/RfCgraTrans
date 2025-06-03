#define S1(k,i)	A[k][i] = i;
#define S2(k,i)	B[k][i] = A[k][i - 1] + A[N - k][i];
#define S3(i)	C[i] = B[0][i];
#define S4(i)	D[i] = C[i - 1];

		int t1, t2, t3, t4;

	register int lbv, ubv;

/* Start of CLooG code */
if (N >= 1) {
  for (t2=0;t2<=N-1;t2++) {
    for (t3=0;t3<=N-1;t3++) {
      S1(t2,t3);
      S2(t2,t3);
    }
  }
  S4(0);
  lbv=1;
  ubv=N-1;
  #pragma ivdep
  #pragma vector always
  for (t2=lbv;t2<=ubv;t2++) {
    S3((t2-1));
    S4(t2);
  }
  S3((N-1));
}
/* End of CLooG code */

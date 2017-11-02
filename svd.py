import numpy as np

if __name__ == "__main__":
    eps = 1.0e-10

    A = np.random.rand(1000,500)*10
    M, N = A.shape

    S = np.zeros(N)
    
    count = 1
    sweep = 0
    sweepmax = 5*N

    tolerance = 10*M*eps

    sweepmax = np.maximum(sweepmax, 12)

    Q = np.identity(N)

    for j in range(N):
        cj = A[:,j]
        sj = np.linalg.norm(cj)
        S[j] = eps*sj

    while(count > 0 and sweep < sweepmax):
        count = N*(N-1)/2

        for j in range(N-1):
            for k in range(j+1,N):
                cj = A[:,j]
                ck = A[:,k]

                p = 2*cj.dot(ck.T)

                a = np.linalg.norm(cj)
                b = np.linalg.norm(ck)
                q = a*a - b*b

                v = np.sqrt(p**2+q**2)

                abserr_a = S[j]
                abserr_b = S[k]

                sort_before = ( a > b )
                orthog = ( abs(p) <= tolerance*a*b )
                noisya = ( a < abserr_a )
                noisyb = ( b < abserr_b )

                if ( sort_before and (orthog or noisya or noisyb)):
                    count -= 1
                    continue

                if ( v == 0 or (not sort_before) ):
                    cosine = 0.0
                    sine = 1.0
                else :
                    cosine = np.sqrt( (v+q)/(2*v) )
                    sine = p / (2*v*cosine)
                
                for i in range(M):
                    Aik = A[i,k]
                    Aij = A[i,j]

                    A[i,j] = Aij*cosine + Aik*sine
                    A[i,k] = -Aij*sine + Aik*cosine

                S[j] = abs(cosine)*abserr_a + abs(sine)*abserr_b
                S[k] = abs(sine)*abserr_a + abs(cosine)*abserr_b

                for i in range(N):
                    Qij = Q[i,j]
                    Qik = Q[i,k]

                    Q[i,j] = Qij*cosine + Qik*sine
                    Q[i,k] = -Qij*sine + Qik*cosine

        sweep += 1

    prev_norm = -1.0

    for j in range(N):
        column = A[:,j]
        norm = np.linalg.norm(column)

        if ( norm == 0.0 or prev_norm == 0.0 or ( j > 0 and norm <= tolerance * prev_norm) ):
            S[j] = 0.0
            A[:j] = [ 0.0 for i in column ]
        else :
            S[j] = norm
            A[:,j] = column / norm
            prev_norm = norm

    print(S)

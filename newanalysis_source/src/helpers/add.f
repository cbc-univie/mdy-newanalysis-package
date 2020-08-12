      SUBROUTINE ZADD(A,B,C,N)
      REAL(KIND=8),INTENT(IN),DIMENSION(:) :: A, B
      REAL(KIND=8),INTENT(INOUT),DIMENSION(:) :: C
      INTEGER :: N

      WRITE(*,'(18X,A)')'ZADD'

      DO J = 1, N
         C(J) = A(J)+B(J)
      ENDDO
      END

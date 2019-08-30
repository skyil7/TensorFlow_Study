# 고수준 API 및 CNN 활용 GAN
한참 달렸으니 이제 복습 차원에서 Part 7에서 배운 GAN을 전에 배운 CNN으로 만들어보자.

## 모르겠는 거
GAN이 X와 Y를 입력받고 X==Y 가 성립함을 학습하는 것으로 이해했는데 아니었나보다.

CNN으로 구성하려니까 2차원 텐서를 사용해야 하는데, dense를 통한 완전 연결 구성은 1차원 텐서밖에 지원을 안한다.

## 발전 방향 - DCGAN
GAN과 CNN의 결합을 위해 GAN과 CNN을 다시금 복습하였다. CNN은 내가 알고 있던 것과는 다르게 다차원 이미지로부터 특징(Feature)을 뽑아내는 신경망이었다.

이러한 CNN의 특징을 GAN에 적용함으로써 더욱 안정적이고 고차원적인 이미지 생성이 가능한데, 이와 관련해서 가장 발전된 연구가 DCGAN이라고 한다.

Deep Convolutional Ganerative Adversarial Network를 학습하기 앞서, CNN과 GAN을 더 잘 익히고 하는 것이 좋을 것 같다.
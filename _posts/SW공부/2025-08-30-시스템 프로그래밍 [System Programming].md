---
layout: post
title: "시스템 프로그래밍_System Programming"
subtitle: Software Runtime System
gh-repo: james-hyeok-kim/james-hyeok-kim.github.io.git
gh-badge: [star, fork, follow]
tags: [System Programming]
comments: true
mathjax: true
author: James Kim
---


[Reference](https://ghi512.tistory.com/10)

# 시스템 프로그램의 종류

1. compilation system (번역 시스템)

2. Operating system (운영체제)

3. Runtime system (런타임 시스템)

4. Hardware consideration (하드웨어 고려)

<img width="757" height="515" alt="image" src="https://github.com/user-attachments/assets/f5fad4bf-aa83-415c-9f74-a37211a0e4fa" />

## 1. Compilation System (번역 시스템)

### 01 언어 계층 구조
번역 시스템의 concept은 언어 계층 구조로, high-level 언어를 기계어로 바꿔주는 시스템이다.

<img width="754" height="530" alt="image" src="https://github.com/user-attachments/assets/58105d71-7337-47b4-9f5b-6ed1d758d41f" />

### 02 전체적인 구조와 6가지 중요 구성요소

<img width="794" height="452" alt="image" src="https://github.com/user-attachments/assets/413f27d1-1681-41c1-bd16-2634515b7b4d" />

1. Editor : 편집기를 사용해서 C 프로그램을 만든다.

- C File : 만들어진 C 프로그램은 input으로 compiler에 들어간다.

2. Compiler : C 프로그램을 어셈블리어로 바꾼다.

- Error Msg : 에러가 있으면 에러 메시지를 출력한다.

- ASM File : 어셈블리어로 바뀐 파일은 input으로 Assembler에 들어간다.

3. Assembler : 어셈블리어를 기계어로 바꾼다.

- Relocatable Object File : 재배치 가능한 오브젝트 파일. Linker에 input으로 들어간다.

4. Linker : 내가 만든 오브젝트 파일과 다른 오브젝트 파일, 라이브러리를 합친다.

- Executable Object File : 수행 가능한 오브젝트 파일. 바이너리 파일(기계어)으로 시스템에 올라가서 수행이 가능하다.

5. Loader : 디스크에 있는 내용을 메모리에 올린다.

- input data → 컴퓨터 프로그램 → results

6. debugger : 수행하고 있는 프로그램의 상태를 체크하고 버그를 잡는다.

### 03 언어 계층 구조와 번역 시스템의 전체적인 구조 사이의 관계

high-level language : C File

assembly language : ASM File

machine language : Relocatable object file, executable object file

< Relocatable object file과 executable object file >  둘 다 binary이다. (이진수로 구성되어 있음)

- Relocatable object file (재배치 가능) : 아직 주소, 외부 참조가 확정되지 않은 상태이다. 즉, 이진수로 만들어지긴 했지만 혼자서는 수행될 수 없고, 언젠가는 재배치되어야 하는 기계어이다.

- executable object file (수행 가능) : 혼자 동적으로 수행이 가능한 기계어이다. 즉, 주소가 결정되어 메인 메모리 상에 어디로 올라갈지도, 외부참조도 결정이 된 상태이다. linking이 완료된 object이다.

### 04 리눅스에서의 예시
* $ : Shell, 명령어 해석기 (Command Interpreter)
* ls : 사용자가 입력한 명령어를 해석해서 처리한다.
* vi : editor, 편집기를 실행한다.
* gcc : compiler, 컴파일-어셈블-링킹을 모두 한꺼번에 수행한다.
* ./a.out : 실제 a.out을 수행한다. (executable object file은 리눅스 상에서 디폴트로 a.out이라는 이름으로 만들어진다.)
* / : 리눅스에서 디렉토리를 구분할 때 사용한다.
* . : 현재 디렉토리

<img width="1280" height="869" alt="image" src="https://github.com/user-attachments/assets/a34d0626-5a0e-4d40-8d0f-774f395eeeeb" />

① vi hello.c : 편집기 실행

① 수행 후 : hello.c 파일 만들어짐

②-1 gcc -S hello.c : 컴파일만 수행 

②-1 수행 후 : hello.s (어셈블리) 파일 만들어짐

②-2 as -o hello.o hello.s : 어셈블러(어셈블리어→기계어)만 수행

②-2 수행 후 : hello.o (재배치 가능한 오브젝트) 파일 만들어짐

②-3 …./collect2/…. : 링커만 수행

②-3 수행 후 : a.out (실행 가능한 오브젝트) 파일 만들어짐

③ ./a.out : a.out 실행

 

Q1) hello.c와 hello.s의 차이점은?

- .c는 high-level 언어이고, .s는 어셈블리 언어이다. 즉, 언어 계층구조의 차이가 있다.

Q2) hello.o와 a.out의 차이점은?

-.o는 재배치 가능한 기계어이고, a.out은 실행 가능한 기계어이다.


## 2. Operating System (운영체제)

### 01 운영체제 내부 구조와 7가지 주요 구성요소

<img width="845" height="534" alt="image" src="https://github.com/user-attachments/assets/d9369fcb-9353-4835-aa19-1b30c4bf3af5" />

* Process Manager : 프로세스 관리
* Virtual Memory : 메모리 관리
* File System : 영속성 제공, 파일 관리
* Device Manager : 장치 관리 (block-HDD, character-키보드, 마우스)
* Network Manager : 통신 프로토콜
* Hardware Interface (HAL; HW Abstraction Layer) : 하드웨어 추상화
* System Call Interface : OS가 사용자 프로그램에게 서비스를 제공할 때 잘 정의된 인터페이스(API)를 제공하는데, 이때의 API를 System Call이라고 부른다. (라이브러리, 프레임워크 등도 API를 제공하는데, 특히 OS가 제공하는 API를 System Call 이라고 부르는 것이다.)

< kernel과 OS >

운영체제(OS)는 Kernel(핵심)이라고도 부른다. 

그런데 사실 OS는 넓은 범위의 SW를 다 포함하는 개념이고, kernel은 그 SW 중에서도 핵심적인 부분, 항상 메모리에 상주해야 하는 부분을 의미한다.

이 커널 부분은 운영체제를 만들 때 크게 만들 수도 있고, 작게 만들 수도 있다. 

리눅스에서는 굉장히 큰 커널을 만들었고, 이렇게 OS를 큰 커널로 만든 경우 OS와 커널 용어를 혼용해서 사용해도 괜찮다. 

그렇기 때문에 리눅스 환경에서는 OS와 커널이 사실상 같은 용어이다. 

(반면, 윈도우 한경에서는 OS 중에서 핵심적인 부분을 커널이라 하고, 그 외의 부분을 다 모아서 통칭 OS라고 한다.)


### 02 하드웨어 구성요소와 운영체제 구조 사이의 관계

운영체제는 자원 관리자라고 불린다.

자원은 물리적 자원(HW component)와 논리적 자원(OS에서 관리하는 자원)으로 나누어지는데, 운영체제는 이 하드웨어 자원들을 논리적 자원들로 추상화시킨다.

* CPU → Process Manager : 운영체제의 추상화를 거쳐 CPU는 프로세스가 된다.
* Main memory → Virtual Memory
* Secondary storage → File system : 컴퓨터 시스템에서 새로운 파일을 하나 만들면 파일은 disk 상에 저장되고, 파일 시스템이라는 OS의 기능을 통해 storage에 저장된다.
* Communication device → Network Manager : 컴퓨터-컴퓨터 간의 대화. 프로토콜을 추상화 시킨 것이 network manager, TCP/IP이다.
* Input device, Output device, Communication device → Device Manager : 장치 관리

### 03 하드웨어 구성요소와 운영체제 구조 사이의 관계

<img width="505" height="269" alt="image" src="https://github.com/user-attachments/assets/682dfea9-cda3-4570-a304-b95a4bc5649c" />

① 초기상태 : Disk, CPU, 메모리 모두 깨끗하다.

---

② 파일을 만든다.

<img width="177" height="181" alt="image" src="https://github.com/user-attachments/assets/1841d460-0784-4841-af4e-115c928ba411" />

#### 👤사용자 관점👤. 편집기(vi)를 열어서 파일(test.c)을 새로 만든다.

<img width="721" height="445" alt="image" src="https://github.com/user-attachments/assets/7a707588-ea3a-4087-9ccd-5d27906ec299" />

#### 🤖시스템 관점🤖 파일이 만들어지면 다음의 내용이 저장되어야 한다.

* 파일의 내용

* 파일에 대한 정보 (만든사람, 날짜, 파일크기, 접근권한 등)
    * inode : 리눅스에서 중요한 자료구조로, ‘파일에 대한 정보’를 담고 있다.
    * disk block : 새로운 파일이 만들어지면 ‘파일의 내용’이 block에 할당되어서 아스키코드에 따라 16진수로 저장된다.
    * 시스템 프로그램 입장에서 disk는 disk block들의 집합이다. 그래서 새로운 파일이 만들어지면 block을 할당해서 그 파일의 내용을 담는다.

 
파일의 내용은 16진수로 저장된다. 

왜일까? 컴퓨터 시스템에서는 데이터를 저장할 때 아스키코드(10진수)를 스고, 컴퓨터에서는 이진수로 저장한다. 

그런데 이 값들을 쭉 이진수로 쓰면 너무 길어지므로 이진수를 4비트끼리 모아서 표시하는데, 이게 바로 16진수이다.

i의 아스키코드 105(10진수) → 69(16진수)

n의 아스키코드 110(10진수) → 6e(16진수)

⭐ POINT ⭐

파일이 만들어지면 그 내용이 disk에 저장되어야 한다. 이 파일의 내용은 disk block이라는 단위로 저장된다.

파일이 만들어지면 파일에 대한 정보도 저장되어야 한다. 이 정보는 inode에 저장된다.

---

③ 파일을 컴파일한다.

<img width="463" height="187" alt="image" src="https://github.com/user-attachments/assets/19ad8280-e9ba-40c5-9883-9467c73bfb39" />

#### 👤사용자 관점 👤

컴파일하면 test.c 파일이 a.out이라는 새로운 이름의 새로운 파일로 만들어진다.

compile 과정 ($gcc) : .c (high-level) → .s (assembly) → .o (relocatable) → a.out (executable)

<img width="517" height="449" alt="image" src="https://github.com/user-attachments/assets/76d97917-0941-4f0c-b245-68618f91f4e0" />

#### 🤖시스템 관점🤖

시스템 프로그램을 배우는 관점에서는 a.out 이라는 새로운 “파일”이 생겼으니 disk 상에서 변화가 있겠구나~ 하고 생각할 수 있다.

- a.out을 관리하기 위한 새로운 inode가 만들어진다.

- disk block을 새로 할당받아서 a.out 파일의 내용을 저장한다. 

Q) test.c → disk block 1개 / a.out → disk block 2개 사용한다. 왜일까?

A) block은 고정된 크기(보통 4KB(4096B))를 가지고 있고, test.c 파일의 크기는 40~50B 정도로 굉장히 작다고 가정한다. 

  char 하나에 1바이트 할당하는데, test.c 파일은 코드를 모두 합쳐서 40-50 char 정도여서 disk block이 하나면 충분하다.
  
  반면, a.out은 컴파일 과정을 거치며 내용이 늘기 때문에 7KB 정도 된다(고 가정한다). 
  
  그래서 disk block 1개(4KB)로는 부족하기 때문에 2개를 할당한다.

⭐ POINT ⭐

새로운 파일이 만들어지면 시스템에서는 그 파일을 관리하기 위한 inode가 1개 만들어진다.

만들어진 파일의 사이즈에 따라 disk block 여러 개를 할당해 내용을 담아둔다.

disk block의 위치 정보는 inode에서 관리한다.

---

④ a.out을 실행한다.

#### 👤사용자 관점👤

모니터에 45가 출력된다.

<img width="737" height="452" alt="image" src="https://github.com/user-attachments/assets/066a85cb-1e6b-4b86-a097-7c2d980f3211" />

#### 🤖시스템 관점🤖

디스크 상에 있는 a.out을 수행하려면 CPU 위로 올려야 한다. 

하지만 CPU는 disk에 바로 접근할 수 없기 때문에 먼저 운영체제가 a.out을 메모리 위에 올려야 한다. 

메모리 내부에서는 페이지들을 할당한다. (메모리는 페이지들의 집합이다.) 

a.out을 올리기 위해서 2개의 page를 할당받고, 각각의 block을 page에 올린다.

⭐ POINT ⭐

a.out을 수행하기 위해 disk block(디스크 상에 있던 데이터)들을 메모리 상으로 올린다.

메모리 상에 데이터를 올리는 공간을 page라고 한다.

<img width="346" height="252" alt="image" src="https://github.com/user-attachments/assets/d392f077-1e46-4900-8ca2-ee17aa508bfa" />

disk block(데이터)을 메모리 상에 올리고서는 process를 만들어줘야 한다. 

프로세스는 수행 중인(살아 움직이는, active object) 프로그램이다. 

process가 가지고 page들이 각각 어디에 있는지 관리할 때에는 page table을 사용한다. 

운영체제는 프로세스를 스케줄링한다. (CPU 할당)

< inode와 page table >

- inode : 파일에 있는 데이터가 어느 disk block에 있는지 관리한다.

- page table : process가 가지고 있는 page들이 어디있는지 정보를 관리한다.

⭐ a.out이 수행되기 위해서는 ⭐

1. disk 상에 있던 a.out이 메모리로 로딩되어야 한다.

2. 새로운 프로세스가 만들어져야 한다.

3.  프로세스가 CPU에 스케쥴링되어야 한다.

<img width="736" height="480" alt="image" src="https://github.com/user-attachments/assets/a4565e53-79a6-4a65-b10b-35214bd6373d" />

추가로 고려해야할 점은 현재 시스템에 내가 수행하는 프로그램(new program)만 있는 것이 아니라, 다른 프로그램들(prev process)이 함께 존재한다는 점이다. 

따라서 다른 프로그램과 CPU를 가지고 경쟁해야 한다. ⇒ 시분할 시스템(time-sharing)으로 구현되어 있다.

< time-sharing system 특징 >


CPU를 골고루 나눠주면서 CPU 이용률을 높인다.

사용자 입장에서는 여러 프로그램들이 동시에 수행되는 것과 같은 추상화를 제공한다.

< process는 동시에 수행되는 것처럼 보인다. >

CPU는 1GHz(10^9Hz)로 동작한다(고 가정하자. 보통 2GHz~3GHz인데 계산의 편의성을 위해). 그리고 요즘 나오는 컴퓨터 시스템은 대부분 한 번의 Hz, 즉 한 번의 clock이 뜰 때마다 한 번의 명령어 수행이 가능하다.

→ 1초에 10^9 클락이 뜬다. 즉, 10^9개의 명령어 수행이 가능하다.

→ 100ms(1/10s) 동안 10^8개의 명령어 수행이 가능하다.

→ 100ms는 사용자 입장에서는 굉장히 짧은 시간이므로 A, B, C 3개의 process가 동시에 수행되는 것처럼 보인다. 하지만 CPU 입장에서는 번갈아 수행되는 것이다. (time-sharing system)

→ process들이 시간을 나눠서 CPU를 공유한다.
 
🤖 a.out 실행 - 시스템 관점 정리 🤖

1. 파일을 만든다

2. 메모리에 올린다

3. CPU에 올린다

4. time-sharing 시스템

 ---

### 04 운영체제에 의해 제공되는 추상화

<img width="882" height="515" alt="image" src="https://github.com/user-attachments/assets/b5dace0d-1f4f-457f-ada7-282ecabfcd7c" />


Process Manager (Task manager) : CPU

Virtual Memory : Main memory

File System : Storage

Device driver : Device

Network protocol : Network




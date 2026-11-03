#include "buzzer.h"

#define Open_song_function 0U       //编译唱歌功能

void buzzer_Init(void)
{
    HAL_TIM_PWM_Start(&htim12, TIM_CHANNEL_2);
    TIM12->CCR2 = 0;               /* 上电先静音 */
}

void buzzer_Open(void)
{
    TIM12->CCR2 = 100;
}

void buzzer_Close(void)
{
    TIM12->CCR2 = 0;
}

//慢慢变响慢慢变弱
void buzzer_PWM(uint8_t high, uint8_t low)
{
    for(int i = low; i < high; i++)
      {
        TIM12->CCR2 = i;
          osDelay(10);
       }
      for(int j = high; j > low; j--)
      {
        TIM12->CCR2 = j;
          osDelay(10);
       }
}

void buzzer_finish_flag(int j)
{
    for (int i = 0; i < j ; i++)
    {
        buzzer_Open();
        osDelay(100);
        buzzer_Close();
    }   
}




#if Open_song_function

//* 特别功能 唱歌 *//

/* ---------- 音符频率表（Hz，以 C4=262 为基准） ---------- */
#define REST   0
#define NOTE_C3  131
#define NOTE_D3  147
#define NOTE_E3  165
#define NOTE_F3  175
#define NOTE_G3  196
#define NOTE_A3  220
#define NOTE_B3  247
#define NOTE_C4  262
#define NOTE_D4  294
#define NOTE_E4  330
#define NOTE_F4  349
#define NOTE_G4  392
#define NOTE_A4  440
#define NOTE_B4  494
#define NOTE_C5  523
#define NOTE_D5  587
#define NOTE_E5  659
#define NOTE_F5  698
#define NOTE_G5  784
#define NOTE_A5  880
#define NOTE_B5  988
#define NOTE_C6  1047

/* ---------- 核心：设频率 ---------- */
void Buzzer_SetFreq(uint32_t freq)
{
    if (freq == 0) {
        TIM12->CCR2 = 0;              /* 静音 */
        return;
    }
    /* 计数时钟固定 1MHz：PSC = 239 (假设 240MHz 输入) */
    uint32_t arr = (1000000U / freq) - 1U;
    if (arr < 2U)     arr = 2U;
    if (arr > 65535U) arr = 65535U;

    TIM12->PSC  = 239;
    TIM12->ARR  = arr;
    TIM12->CCR2 = arr / 2;            /* 50% 占空比 */
    /* 产生一次更新事件让 PSC/ARR 立即生效，避免第一个音符走音 */
    TIM12->EGR = TIM_EGR_UG;
}

/* ---------- 《小星星》 ---------- */
const uint16_t twinkle_notes[] = {
    /* Do Do Sol Sol La La Sol */
    NOTE_C4, NOTE_C4, NOTE_G4, NOTE_G4, NOTE_A4, NOTE_A4, NOTE_G4,
    /* Fa Fa Mi Mi Re Re Do */
    NOTE_F4, NOTE_F4, NOTE_E4, NOTE_E4, NOTE_D4, NOTE_D4, NOTE_C4,

    /* Do Do Sol Sol La La Sol */
    NOTE_C4, NOTE_C4, NOTE_G4, NOTE_G4, NOTE_A4, NOTE_A4, NOTE_G4,
    /* Fa Fa Mi Mi Re Re Do */
    NOTE_F4, NOTE_F4, NOTE_E4, NOTE_E4, NOTE_D4, NOTE_D4, NOTE_C4,

    /* Sol Sol Fa Fa Mi Mi Re */
    NOTE_G4, NOTE_G4, NOTE_F4, NOTE_F4, NOTE_E4, NOTE_E4, NOTE_D4,
    /* Sol Sol Fa Fa Mi Mi Re */
    NOTE_G4, NOTE_G4, NOTE_F4, NOTE_F4, NOTE_E4, NOTE_E4, NOTE_D4,

    /* Do Do Sol Sol La La Sol */
    NOTE_C4, NOTE_C4, NOTE_G4, NOTE_G4, NOTE_A4, NOTE_A4, NOTE_G4,
    /* Fa Fa Mi Mi Re Re Do */
    NOTE_F4, NOTE_F4, NOTE_E4, NOTE_E4, NOTE_D4, NOTE_D4, NOTE_C4,
};

/* 时长（ms），四分音符 = 400ms，二分音符 = 800ms */
const uint16_t twinkle_durs[] = {
    400, 400, 400, 400, 400, 400, 800,
    400, 400, 400, 400, 400, 400, 800,

    400, 400, 400, 400, 400, 400, 800,
    400, 400, 400, 400, 400, 400, 800,

    400, 400, 400, 400, 400, 400, 800,
    400, 400, 400, 400, 400, 400, 800,

    400, 400, 400, 400, 400, 400, 800,
    400, 400, 400, 400, 400, 400, 800,
};

#define TWINKLE_LEN (sizeof(twinkle_notes) / sizeof(twinkle_notes[0]))



/* ---------- 《Love Song》主歌片段 音符频率 (Hz) ---------- */
/* 1=C4(262), 2=D4(294), 3=E4(330), 4=F4(349), 5=G4(392), 6=A4(440) */
const uint16_t lovesong_notes[] = {
    /* 0176 | 532- */
    REST, NOTE_C4, NOTE_B3, NOTE_A3,   // 0 1 7 6 (7是B3, 低音B)
    NOTE_G4, NOTE_E4, NOTE_D4, NOTE_D4, // 5 3 2 - (2延长)

    /* 0112 | 355- */
    REST, NOTE_C4, NOTE_C4, NOTE_D4,   // 0 1 1 2
    NOTE_E4, NOTE_G4, NOTE_G4, NOTE_G4, // 3 5 5 - (5延长)

    /* 0176 | 5321 */
    REST, NOTE_C4, NOTE_B3, NOTE_A3,   // 0 1 7 6
    NOTE_G4, NOTE_E4, NOTE_D4, NOTE_C4, // 5 3 2 1

    /* 1231 | 1066553 */
    NOTE_C4, NOTE_D4, NOTE_E4, NOTE_C4, // 1 2 3 1
    NOTE_C4, REST, NOTE_A4, NOTE_A4,    // 1 0 6 6
    NOTE_G4, NOTE_G4, NOTE_E4, NOTE_G4  // 5 5 3 (最后接5)
};

/* 对应的时长 (ms) */
/* 假设速度约 110bpm，四分音符 ≈ 545ms。这里取整，用 500ms 作为基准拍 */
#define LOVESONG_TEMPO_MS 500

const uint16_t lovesong_durs[] = {
    /* 0176 | 532- */
    250, 250, 250, 250,                // 0 1 7 6 (八分音符为主)
    LOVESONG_TEMPO_MS, 250, 250, LOVESONG_TEMPO_MS*2, // 5 3 2 - (2延长一拍)

    /* 0112 | 355- */
    250, 250, 250, 250,
    LOVESONG_TEMPO_MS, 250, 250, LOVESONG_TEMPO_MS*2,

    /* 0176 | 5321 */
    250, 250, 250, 250,
    LOVESONG_TEMPO_MS, 250, 250, 250, // 5 3 2 1 (1是八分)

    /* 1231 | 1066553 */
    250, 250, 250, 250,                // 1 2 3 1
    LOVESONG_TEMPO_MS, 250, 250, 250,  // 1 0 6 6
    250, 250, 250, 250                 // 5 5 3 (接5)
};

#define LOVESONG_LEN (sizeof(lovesong_notes) / sizeof(lovesong_notes[0]))



/* ---------- 《青花瓷》副歌片段 音符频率 ---------- */
/* 基于 Arduino 蜂鸣器代码的频率数组转换而来 */
const uint16_t qinghuaci_notes[] = {
    NOTE_E5, NOTE_D5, NOTE_E5, NOTE_F5, NOTE_E5, NOTE_D5, NOTE_B4, NOTE_C5,
    NOTE_A4, NOTE_B4, NOTE_C5, NOTE_D5, NOTE_B4, NOTE_C5, NOTE_E5, NOTE_D5,
    NOTE_E5, NOTE_F5, NOTE_E5, NOTE_D5, NOTE_B4, NOTE_C5, NOTE_A4, NOTE_B4,
    NOTE_C5, NOTE_D5, NOTE_E5, NOTE_A5, NOTE_G5, NOTE_C5, NOTE_E5, NOTE_D5,
    NOTE_B4, NOTE_C5
};

/* ---------- 对应的时长 (ms) ---------- */
/* 原代码用 1,2,3,4 表示节拍倍数，这里设定 1 个单位 = 250ms，约 120bpm 的听感 */
#define QINGHUACI_BEAT_MS 250

const uint16_t qinghuaci_durs[] = {
    /* 4,4,2,2,2,1,3,1 */
    QINGHUACI_BEAT_MS*4, QINGHUACI_BEAT_MS*4, QINGHUACI_BEAT_MS*2, QINGHUACI_BEAT_MS*2,
    QINGHUACI_BEAT_MS*2, QINGHUACI_BEAT_MS,   QINGHUACI_BEAT_MS*3, QINGHUACI_BEAT_MS,
    /* 4,4,2,2,2,1,4,4 */
    QINGHUACI_BEAT_MS*4, QINGHUACI_BEAT_MS*4, QINGHUACI_BEAT_MS*2, QINGHUACI_BEAT_MS*2,
    QINGHUACI_BEAT_MS*2, QINGHUACI_BEAT_MS,   QINGHUACI_BEAT_MS*4, QINGHUACI_BEAT_MS*4,
    /* 2,2,2,1,3,1,4,4 */
    QINGHUACI_BEAT_MS*2, QINGHUACI_BEAT_MS*2, QINGHUACI_BEAT_MS*2, QINGHUACI_BEAT_MS,
    QINGHUACI_BEAT_MS*3, QINGHUACI_BEAT_MS,   QINGHUACI_BEAT_MS*4, QINGHUACI_BEAT_MS*4,
    /* 2,2,4,4,4,3,1,4,4,2,2 */
    QINGHUACI_BEAT_MS*2, QINGHUACI_BEAT_MS*2, QINGHUACI_BEAT_MS*4, QINGHUACI_BEAT_MS*4,
    QINGHUACI_BEAT_MS*4, QINGHUACI_BEAT_MS*3, QINGHUACI_BEAT_MS,   QINGHUACI_BEAT_MS*4,
    QINGHUACI_BEAT_MS*4, QINGHUACI_BEAT_MS*2, QINGHUACI_BEAT_MS*2
};

#define QINGHUACI_LEN (sizeof(qinghuaci_notes) / sizeof(qinghuaci_notes[0]))



/* ---------- 《大悲咒》片段（基于光遇号角谱，C调） ---------- */
const uint16_t dabeizhou_notes[] = {
    // 1 2 2 2 2 2 3 3 5 3 3 3 2 1
    NOTE_C4, NOTE_D4, NOTE_D4, NOTE_D4, NOTE_D4, NOTE_D4, NOTE_E4, NOTE_E4,
    NOTE_G4, NOTE_E4, NOTE_E4, NOTE_E4, NOTE_D4, NOTE_C4,

    // 1 3 2 2 2 2 1 6 6 2 2 2 2 2 2 3 3 5
    NOTE_C4, NOTE_E4, NOTE_D4, NOTE_D4, NOTE_D4, NOTE_D4, NOTE_C4, NOTE_A3,
    NOTE_A3, NOTE_D4, NOTE_D4, NOTE_D4, NOTE_D4, NOTE_D4, NOTE_D4, NOTE_E4,
    NOTE_E4, NOTE_G4,

    // 3 2 1 1 1 3 2 2 2 2 2
    NOTE_E4, NOTE_D4, NOTE_C4, NOTE_C4, NOTE_C4, NOTE_E4, NOTE_D4, NOTE_D4,
    NOTE_D4, NOTE_D4, NOTE_D4
};

/* 时值：这里全部给 300ms，你可以根据听感调整某些音符的长短 */
#define DABEIZHOU_BEAT_MS 300
const uint16_t dabeizhou_durs[] = {
    300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300,
    300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300,
    300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300
};

#define DABEIZHOU_LEN (sizeof(dabeizhou_notes) / sizeof(dabeizhou_notes[0]))




void Buzzer_PlayMelody(const uint16_t *notes, const uint16_t *durs,
                       uint16_t len, uint8_t loop)
{
    do {
        for (uint16_t i = 0; i < len; i++) {
            Buzzer_SetFreq(notes[i]);
            osDelay(durs[i]);
            Buzzer_SetFreq(0);           /* 音符之间留一点静音，节拍更清晰 */
            osDelay(20);                 /* 20ms 小间隔，可按喜好调 */
        }
        Buzzer_SetFreq(0);
    } while (loop);
}

void Buzzer_ChooseSong(uint8_t i)
{
    switch(i)
    {
        case 1:
            Buzzer_PlayMelody(twinkle_notes, twinkle_durs, TWINKLE_LEN, 0); // 小星星    
            break;
        case 2:
            Buzzer_PlayMelody(lovesong_notes, lovesong_durs, LOVESONG_LEN, 0);// love song
            break;
        case 3:
            Buzzer_PlayMelody(qinghuaci_notes, qinghuaci_durs, QINGHUACI_LEN, 0);   //青花瓷
            break;
        case 4: // 大悲咒 片段
            Buzzer_PlayMelody(dabeizhou_notes, dabeizhou_durs, DABEIZHOU_LEN, 0);
            break;
        default:
            break;
    }  

}

#endif

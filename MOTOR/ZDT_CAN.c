#include "ZDT_CAN.h"

#define CAN_TX_TIMEOUT_MS   20U

static uint8_t s_can_initialized = 0U;
static CAN_ExtRxCallback_t s_extRxCb = NULL;

/* ---------- DLC 辅助 ---------- */
static uint32_t CAN_BytesToDLC(uint8_t bytes)
{
    if (bytes > 8U) bytes = 8U;
    switch (bytes)
    {
        case 0U: return FDCAN_DLC_BYTES_0;
        case 1U: return FDCAN_DLC_BYTES_1;
        case 2U: return FDCAN_DLC_BYTES_2;
        case 3U: return FDCAN_DLC_BYTES_3;
        case 4U: return FDCAN_DLC_BYTES_4;
        case 5U: return FDCAN_DLC_BYTES_5;
        case 6U: return FDCAN_DLC_BYTES_6;
        case 7U: return FDCAN_DLC_BYTES_7;
        default: return FDCAN_DLC_BYTES_8;
    }
}

static uint8_t CAN_DLCToBytes(uint32_t dlc)
{
    switch (dlc)
    {
        case FDCAN_DLC_BYTES_0: return 0U;
        case FDCAN_DLC_BYTES_1: return 1U;
        case FDCAN_DLC_BYTES_2: return 2U;
        case FDCAN_DLC_BYTES_3: return 3U;
        case FDCAN_DLC_BYTES_4: return 4U;
        case FDCAN_DLC_BYTES_5: return 5U;
        case FDCAN_DLC_BYTES_6: return 6U;
        case FDCAN_DLC_BYTES_7: return 7U;
        case FDCAN_DLC_BYTES_8: return 8U;
        default: return 8U;
    }
}

void CAN_RegisterExtRxCallback(CAN_ExtRxCallback_t cb)
{
    s_extRxCb = cb;
}

/* ---------- 初始化 ---------- */
void USER_CAN1_Init(void)
{
    if (s_can_initialized != 0U) return;

    FDCAN_FilterTypeDef sFilterConfig = {0};

    /* 1. 配置扩展帧全接收 (Filter 0) */
    sFilterConfig.IdType = FDCAN_EXTENDED_ID;
    sFilterConfig.FilterIndex = 0U;
    sFilterConfig.FilterType = FDCAN_FILTER_MASK;
    sFilterConfig.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
    sFilterConfig.FilterID1 = 0x00000000U;
    sFilterConfig.FilterID2 = 0x00000000U;
    if (HAL_FDCAN_ConfigFilter(&hfdcan1, &sFilterConfig) != HAL_OK) Error_Handler();

    /* 2. 配置标准帧全接收 (Filter 0，标准帧和扩展帧索引各自独立) */
    sFilterConfig.IdType = FDCAN_STANDARD_ID;
    sFilterConfig.FilterIndex = 0U;
    sFilterConfig.FilterType = FDCAN_FILTER_MASK;
    sFilterConfig.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
    sFilterConfig.FilterID1 = 0x00000000U;
    sFilterConfig.FilterID2 = 0x00000000U;
    if (HAL_FDCAN_ConfigFilter(&hfdcan1, &sFilterConfig) != HAL_OK) Error_Handler();

    /* 非匹配帧全部拒绝 */
    if (HAL_FDCAN_ConfigGlobalFilter(&hfdcan1,
                                     FDCAN_REJECT, FDCAN_REJECT,
                                     FDCAN_REJECT_REMOTE, FDCAN_REJECT_REMOTE) != HAL_OK)
        Error_Handler();

    if (HAL_FDCAN_Start(&hfdcan1) != HAL_OK)
        Error_Handler();

    if (HAL_FDCAN_ActivateNotification(&hfdcan1,
            FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0) != HAL_OK)
        Error_Handler();

    s_can_initialized = 1U;
}

/* ---------- 发送 ---------- */
void can_SendCmd(__IO uint8_t *cmd, uint8_t len)
{
    uint8_t payloadIndex = 0U;
    uint8_t packetIndex  = 0U;
    uint8_t payloadLength;

    if ((cmd == NULL) || (len < 3U)) return;

    payloadLength = (uint8_t)(len - 2U);

    while (payloadIndex < payloadLength)
    {
        FDCAN_TxHeaderTypeDef txHeader = {0};
        uint8_t txData[8] = {0};
        uint8_t remaining = (uint8_t)(payloadLength - payloadIndex);
        uint8_t bytesThisPacket = (remaining > 7U) ? 7U : remaining;
        uint32_t tickStart;
        HAL_StatusTypeDef status;

        txHeader.Identifier = ((uint32_t)cmd[0] << 8U) | packetIndex;
        txHeader.IdType = FDCAN_EXTENDED_ID;
        txHeader.TxFrameType = FDCAN_DATA_FRAME;
        txHeader.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
        txHeader.BitRateSwitch = FDCAN_BRS_OFF;
        txHeader.FDFormat = FDCAN_CLASSIC_CAN;
        txHeader.TxEventFifoControl = FDCAN_NO_TX_EVENTS;
        txHeader.MessageMarker = packetIndex;
        txHeader.DataLength = CAN_BytesToDLC((uint8_t)(bytesThisPacket + 1U));

        txData[0] = cmd[1];
        for (uint8_t i = 0U; i < bytesThisPacket; i++)
            txData[i + 1U] = cmd[payloadIndex + 2U + i];

        tickStart = HAL_GetTick();
        do
        {
            status = HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan1, &txHeader, txData);
            if (status == HAL_OK) break;
        }
        while ((HAL_GetTick() - tickStart) < CAN_TX_TIMEOUT_MS);

        if (status != HAL_OK) return;

        payloadIndex = (uint8_t)(payloadIndex + bytesThisPacket);
        packetIndex++;
    }
}

/* ---------- RX 回调 ---------- */
void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *hfdcan, uint32_t RxFifo0ITs)
{
    if (hfdcan != &hfdcan1) return;
    if ((RxFifo0ITs & FDCAN_IT_RX_FIFO0_NEW_MESSAGE) == 0U) return;

    while (HAL_FDCAN_GetRxFifoFillLevel(hfdcan, FDCAN_RX_FIFO0) > 0U)
    {
        FDCAN_RxHeaderTypeDef rxHeader;
        uint8_t rxData[8] = {0};
        uint8_t dataLength;

        if (HAL_FDCAN_GetRxMessage(hfdcan, FDCAN_RX_FIFO0,
                                   &rxHeader, rxData) != HAL_OK)
            return;

        dataLength = CAN_DLCToBytes(rxHeader.DataLength);
        if (dataLength == 0U) continue;

        // 关键修改：不管是标准帧还是扩展帧，通通传出去！
        if (s_extRxCb != NULL) {
            s_extRxCb(rxHeader.Identifier, rxData, dataLength);
        }
//        if (rxHeader.IdType == FDCAN_EXTENDED_ID)
//        {
//            if (s_extRxCb != NULL)
//                s_extRxCb(rxHeader.Identifier, rxData, dataLength);
//        }
        /* 标准帧直接忽略，因为不接达妙 */
    }

}

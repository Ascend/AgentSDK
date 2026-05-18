import asyncio
import time
import os
from loguru import logger

from transformers import AutoTokenizer

from agents.agents import DTNAgent
from aura.runner.agent_engine_wrapper.rllm.agent_execution_engine import AgentExecutionEngine
from agents.dtn_agent.environment.dtn_env import DTNEnvironment
from agents.rewards import dtn_reward_fn


def save_trajectory(trajectory):
    import pickle
    import os
    from datetime import datetime

    output_path = r"/data/c30040028/codes/AgenticRL/output"
    file_path = os.path.join(output_path, f'trajectory_{datetime.strftime(datetime.now(), "%m%d%H%M")}.pkl')

    with open(file_path, 'wb') as f:
        pickle.dump(trajectory, f)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    n_parallel_agents = 8
    model_name = "/opt/models/Qwen_QWQ-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    agent_args = {"parser_name": "dtn"}
    env_args = {
        "reward_fn": dtn_reward_fn,
    }
    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=DTNAgent,
        agent_args=agent_args,
        env_class=DTNEnvironment,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={"base_url": "http://10.50.112.169:8000/v1", "api_key": "None"},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=1024 * 96,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
        max_steps=10,
    )

    tasks = [
        {
            'data_source': 'ioh',
            'question': '故障工单信息如下：\nTT Title: 14 NE DOWN UNDER HUB SITE 13KRW0426_KEDAWUNG_KRW_PL, MC-CIKAMPEK, JAYA\nTT Number: INC-20250227-00015958\n*Possibility Severity Level:* CRITICAL\nSTART: (2025-02-27 13:16:02)  - End: (-)\nDuration: -\nMC Impacted: MC-KETAPANG\n\n*▪Service Impact:*\n▪Site Down:\n2G = 7 BTS\n4G = 7 eNodeB\n\n*▪MC Cluster Impacted:*\nMC-CIKAMPEK\n\n*▪Site Impact:*\n13KRW0609\n13KRW0607\n13KRW0615\n13KRW0426\n13KRW0616\n13KRW0425\n13KRW0475',
            'ground_truth': '接入环双点开环造成上游传输路径中断，引发多站点断站：\n故障点1：\n根因描述：\n2025/2/26  0:29 Site 13KRW0475路由器JKT-PTDA-OPT-H910D和Site 13KRW0497路由器JKT-PGBR-EN1-H8M08之间的物理端口down，引发链路中断。\n根因告警：\nPhysical Port Down;JKT-PTDA-OPT-H910D_GigabitEthernet0/2/0\nPhysical Port Down;JKT-PGBR-EN1-H8M08_GigabitEthernet0/1/3\n\n故障点2：\n根因描述：\n2025/2/27 10:54 Site 13KRW0426发生市电故障，2025/2/27  13:08站点供电不足引发路由器下电，传输路径中断；\n根因告警：\nMains Failure;KEDAWUNG_KRW_PL\n\n故障点3：\n根因描述：\n2025/2/23  10:23 Site 13KRW0475路由器JKT-PTDA-OPT-H910D连接到无线基站的端口发生物理端口down，引发Site 13KRW0475无线基站退服。\n根因告警：\nPhysical Port Down;JKT-PTDA-OPT-H910D_GigabitEthernet0/2/1',
        },
        {
            'data_source': 'ioh',
            'question': '故障工单信息如下：\nTT Title:  6 SITES DOWN UNDER MC-CENGKARENG\nTT Number: INC-20250524-00003013\n*Possibility Severity Level:* CRITICAL\nSTART: (24-May-2025 04:13)  - End: (-)\nDuration: -\nMC Impacted: MC-KETAPANG\n\nAction Taken – Restoration Key Activities:\n24-May-2025 04:13 DIOC FORAN observed many sites down under MC-CENGKARENG\n\n*On Duty Team :*\nIM/RL : Bagus / Julreza/Marlan\nSHIFT MANAGER :  S Fikri/Hendra CG\nFSO:\nOSP :\n2025-05-24 04:13 12JKB0062\n2025-05-24 04:13 12JKB0096\n2025-05-24 04:13 12JKU0570\n2025-05-24 04:13 12JKB0076\n2025-05-24 04:13 11TGN0738\n2025-05-24 04:13 12JKB0043',
            'ground_truth': '接入环双点开环造成上游传输路径中断，引发多站点断站：\n故障点1：\n根因描述：\nSite 12JKB0076路由器JKT-TMNS-EN1-H910D和Site 12JKB0036路由器JKT-PTAU-EN1-H910D之间的物理链路在2025-05-24 01:23中断；\n根因告警：\nPhysical Port Down;JKT-TMNS-EN1-H910D_100GE0/2/25\nPhysical Port Down;JKT-PTAU-EN1-H910D_100GE0/2/24\n\n故障点1：\n根因描述：\nSite 12JKB0096在2025-05-24 04:10发生市电故障，引发无线和传输设备供电不足，业务连接中断；\n根因告警：\nMains Failure;GT_MENCENG_EP',
        },
        {
            'data_source': 'ioh',
            'question': '故障工单信息如下：\nTitle:  26 SITES DOWN UNDER MC-KOLAKA, MC-SOUTH KONAWE, KALISUMAPA\nTT Number: INC-20250525-00021447\n*Possibility Severity Level:* CRITICAL\nSTART: (25-May-2025 20:36)  - End: (-)\nDuration: -\nMC Impacted: MC-KOLAKA, MC-SOUTH KONAWE\nService Impacted: 2G: 26 BTS, 4G: 26 eNodeB\n\nAction Taken – Restoration Key Activities:\n25-May-2025 20:36 DIOC FORAN observed many sites down under MC-KOLAKA, MC-SOUTH KONAWE, Circle KALISUMAPA\n\n*On Duty Team :*\nIM/RL :  Bagus / Julreza/Marlan\nSHIFT MANAGER :  S Fikri/Hendra CG\nFSO:\nOSP : \n\n5/25/2025 20:36 30BBN0009\n5/25/2025 20:36 30KKA0024\n5/25/2025 20:36 30KKA0016\n5/25/2025 20:36 30KKA0020\n5/25/2025 20:36 30KKA0050\n5/25/2025 20:36 30KKA0051\n5/25/2025 20:36 30KKA0048\n5/25/2025 20:36 30KKA0044\n5/25/2025 20:36 30KKA0015\n5/25/2025 20:36 30KKA0053\n5/25/2025 20:36 30KKA0077\n5/25/2025 20:36 30KKA0076\n5/25/2025 20:36 30BBN0001\n5/25/2025 20:36 30BBN0011\n5/25/2025 20:36 30KKA0072\n5/25/2025 20:36 30KKA0014\n5/25/2025 20:36 30BBN0017\n5/25/2025 20:36 30KKA0041\n5/25/2025 20:36 30BBN0014\n5/25/2025 20:36 30KKA0040\n5/25/2025 20:36 30KKA0023\n5/25/2025 20:36 30KKA0019\n5/25/2025 20:36 30BBN0016\n5/25/2025 20:36 30KKA0039\n5/25/2025 20:36 30BBN0015\n5/25/2025 20:36 30BBN0012',
            'ground_truth': 'Site 30KKA0020至核心网的两条传输链路先后发生中断，导致传输路径中断，下游批量基站退服。\n故障点1：\n根因描述：\nSite 30KKA0020和Site 30KKA0025间通过微波链路承载，2025-05-16 12:58:21微波设备断连，站点间微波路径中断；\n根因告警：\nLDP neighbour down alarm;KND-KLK-AN1-ZM8S\nISIS neighbour down alarm;KND-KLK-AN1-ZM8S\n\n故障点2：\n根因描述：Site 30KKA0020路由器KND-OOKL-EN1-Z20HS和Site 30KKA0028路由器KND-PLAA-EN1-Z20HS之间的光纤链路在2025-05-25 20:34:21中断。\n根因告警：\nEthernet Physical (ETPI) Port down;KND-OOKL-EN1-Z20HS_xxvgei-1/1/0/28\nEthernet Physical (ETPI) Port down;KND-PLAA-EN1-Z20HS_xxvgei-1/1/0/28\n\n',
        },
        {
            'data_source': 'ioh',
            'question': '故障工单信息如下：\nTT Title:  15 SITES DOWN UNDER MC-KETAPANG, KALISUMAPA\nTT Number: INC-20250525-00008328\n*Possibility Severity Level:* CRITICAL\nSTART: (25-May-2025 09:56)  - End: (-)\nDuration: -\nMC Impacted: MC-KETAPANG\nService Impacted: 2G: 15 BTS, 4G: 15 eNodeB\n\nAction Taken – Restoration Key Activities:\n25-May-2025 09:56 DIOC FORAN observed many sites down under MC-KETAPANG, Circle KALISUMAPA\n\n*On Duty Team :*\nIM/RL : Bagus / Julreza/Marlan\nSHIFT MANAGER :  S Fikri/Hendra CG\nFSO:\nOSP :\n5/25/2025 9:56 20KTP0109\n5/25/2025 9:56 20KTP0113\n5/25/2025 9:56 20KTP0130\n5/25/2025 9:56 20KTP0117\n5/25/2025 9:56 20KTP0107\n5/25/2025 9:56 20KTP0106\n5/25/2025 9:56 20KTP0108\n5/25/2025 9:56 20KTP0115\n5/25/2025 9:56 20KTP0126\n5/25/2025 9:56 20KTP0145\n5/25/2025 9:56 20KTP0116\n5/25/2025 9:56 20KTP0111\n5/25/2025 9:56 20KTP0112\n5/25/2025 9:56 20KTP0114\n5/25/2025 9:55 20KTP0067',
            'ground_truth': '故障点1：\n根因描述：\nHub站点20KTP0109市电长时间中断，油机每运行5小时后停机1小时，停机期间站点蓄电池供电异常，仅能维持约15分钟供电，约15分钟后供电不足微波设备下线，传输链路中断，进而引发下游站点断站。\n根因告警：\nGenset Running@GLTE_KLUWIN_EP\nBASE STATION INFORMATION-Low PDU input voltage@GLTE_KLUWIN_EP',
        },
        {
            'data_source': 'ioh',
            'question': '故障工单信息如下：\nTT Title:  4 SITES DOWN UNDER MC-KETAPANG, KALISUMAPA\nTT Number: INC-20250227-00013537\n*Possibility Severity Level:* CRITICAL\nSTART: (27-Feb-2025 12:16)  - End: (-)\nDuration: -\nMC Impacted: MC-KETAPANG\nService Impacted: 2G: 4 BTS, 4G: 4 eNodeB\n\nAction Taken – Restoration Key Activities:\n27-Feb-2025 12:16 DIOC FORAN observed many sites down under MC-KETAPANG\n\n*On Duty Team :*\nIM/RL : Bagus / Julreza/Marlan\nSHIFT MANAGER :  S Fikri/Hendra CG\nFSO:\nOSP :\n2/27/2025 12:16 30BBN0011\n2/27/2025 12:16 30KKA0012\n2/27/2025 12:16 30KKA0009\n2/27/2025 12:16 30KKA0014\n2/27/2025 12:16 20KTP0107',
            'ground_truth': '故障点1：\n根因描述：\nSite 30BBN0011市电中断，导致站上微波设备下线，引发下游批量站点退服。\n根因告警：\nPMS Communication Failure;30BBN0011_TEPPOE_TB\nNE_COMMU_BREAK;30BBN0011-TEPPOE_TB-1',
        },
        {
            'data_source': 'ioh',
            'question': '故障工单信息如下：\nTitle:  25 SITE DOWN UNDER MC-BUKITTINGGI\nMC-PASAMAN BARAT\nMC-SOLOK, SUMATERA"\nTT Number: INC-20250524-00009762\n*Possibility Severity Level:* CRITICAL\nSTART: (24-May-2025 10:08)  - End: (-)\nDuration: -\n"MC Impacted: MC-BUKITTINGGI\nMC-PASAMAN BARAT\nMC-SOLOK"\n\nAction Taken – Restoration Key Activities:\n"24-May-2025 10:08 DIOC FORAN observed many sites down under MC-BUKITTINGGI\nMC-PASAMAN BARAT\nMC-SOLOK, Circle SUMATERA"\n\n*On Duty Team :*\nIM/RL :  Bagus / Julreza/Marlan\nSHIFT MANAGER :  S Fikri/Raffi\nFSO:\nOSP : \n\n\nMC Cluster\nMC-BUKITTINGGI\nMC-PASAMAN BARAT\nMC-SOLOK\n\nLast Occurred On Site ID\n24/05/2025 10:09 03LBU0022\n24/05/2025 10:08 03LBS0004\n24/05/2025 10:08 03LBS0022\n24/05/2025 10:08 03LBS0015\n24/05/2025 10:08 03LBS0017\n24/05/2025 10:08 03LBS0031\n24/05/2025 10:08 03LBS0027\n24/05/2025 10:08 03LBS0029\n24/05/2025 10:08 03LBS0025\n24/05/2025 10:08 03LBS0001\n24/05/2025 10:08 03LBS0012\n24/05/2025 10:08 03LBS0005\n24/05/2025 10:08 03LBS0020\n24/05/2025 10:08 03LBS0019\n24/05/2025 10:08 03LBS0023\n24/05/2025 10:08 03LBS0021\n24/05/2025 10:08 03LBS0009\n24/05/2025 10:08 03LBS0014\n24/05/2025 10:08 03LBS0003\n24/05/2025 10:08 03LBS0028\n24/05/2025 10:08 03LBS0036\n24/05/2025 10:08 03SPA0059\n24/05/2025 10:08 03SPA0071\n24/05/2025 10:08 03SPA0074\n24/05/2025 10:07 03ASK0033\n',
            'ground_truth': '上游多条传输链路故障，导致至核心网的传输路径中断，造成批量站点退服：\n故障点1：\n根因描述：\n2025-05-24 10:07:20 Site 03SPA0043路由器PAD-SMPE-EN1-Z20HS和Site 03LBS0005路由器PAD-PNTI-EN1-Z20HS间物理连接中断，引发传输中断；\n根因告警：\nEthernet Physical (ETPI) Port down;PAD-SMPE-EN1-Z20HS_xxvgei-1/1/0/20\nEthernet Physical (ETPI) Port down;PAD-PNTI-EN1-Z20HS_xxvgei-1/1/0/28\n\n故障点2：\n根因描述：\n2025-05-23 17:05:08 Site 03LBS0026路由器PAD-NJKO-EN1-Z20HS和Site 03LBS0016路由器PAD-ULAN-EN1-Z20HS间物理连接中断，引发传输中断；\n根因告警：\nEthernet Physical (ETPI) Port down;PAD-ULAN-EN1-Z20HS_xxvgei-1/1/0/28\nEthernet Physical (ETPI) Port down;PAD-NJKO-EN1-Z20HS_xxvgei-1/1/0/28\n\n故障点3：\n根因描述：\nSite 03LBS0017路由器PAD-BNJL-EN1-Z20HS和Site 03BTT0019路由器PAD-AUK-EN1-Z20HS间路由转发异常，原因：Area 25 的 BGP 路由没有通过 ABR 正确传播到OSPF Area 20，导致 BGP 路由在 Area 25 中被隔离，需要在 BCP-CN1 上添加 BGP 对等体，以恢复 BGP 路由连接到OSPF Area 20。\n根因告警：\nBGP neighbour down alarm;PAD-BNJL-EN1-Z20HS',
        },
        {
            'data_source': 'ioh',
            'question': '故障工单信息如下：\nTitle:  14 SITE DOWN UNDER MC-SAMPIT, KALISUMAPA\nTT Number: INC-20250525-00014177\n*Possibility Severity Level:* CRITICAL\nSTART: (25-May-2025 13:54)  - End: (-)\nDuration: -\nMC Impacted: MC-SAMPIT\nService Impacted: 2G: 14 BTS, 4G: 14 eNodeB\n*Findings:*\nOn query\n\n*Reason For Outage :*\nUnder Investigate\n\nAction Taken – Restoration Key Activities:\n25-May-2025 13:54 DIOC FORAN observed many sites down under MC-SAMPIT, Circle KALISUMAPA\n\nLast Occurred On Site ID\n5/25/2025 13:54 21SPT0054\n5/25/2025 13:54 21SPT0037\n5/25/2025 13:54 21SPT0052\n5/25/2025 13:54 21SPT0053\n5/25/2025 13:54 21SPT0201\n5/25/2025 13:54 21SPT0022\n5/25/2025 13:54 21SPT0058\n5/25/2025 13:54 21SPT0016\n5/25/2025 13:54 21SPT0014\n5/25/2025 13:54 21SPT0051\n5/25/2025 13:54 21SPT0021\n5/25/2025 13:54 21SPT0043\n5/25/2025 13:54 21SPT0048\n5/25/2025 13:54 21PBU014',
            'ground_truth': '上游传输链路先后故障，导致至核心网的传输路径中断，造成批量站点退服：\n故障点1：\n根因描述：\n2025-05-25 13:07:00 Site21SPT0065路由器PLK-KULY-EN1-Z20HS和Site 21PBU0140路由器PLK-ORAG-EN1-Z20HS间物理连接中断，引发传输中断；\n根因告警：\nEthernet Physical (ETPI) Port down;PLK-KULY-EN1-Z20HS_cgei-1/1/0/33\nEthernet Physical (ETPI) Port down;PLK-ORAG-EN1-Z20HS_cgei-1/1/0/33\n\n故障点2：\n根因描述：\n2025-05-25 13:54:19 Site 18SPT0915路由器PLK-SUBG-EN1-Z20HS和Site 18SPT007路由器PLK-SBK-EN1-Z20HS间物理连接中断，引发传输中断；\n根因告警：\nEthernet Physical (ETPI) Port down;PLK-SUBG-EN1-Z20HS_xxvgei-1/1/0/28\nEthernet Physical (ETPI) Port down;PLK-SBK-EN1-Z20HS_xxvgei-1/1/0/32',
        },
        {
            'data_source': 'ioh',
            'question': '故障工单信息如下：\nTT Title: 5 SITES DOWN UNDER SITE 16PMN0015_TEJA_BARAT_PAMEKASAN_PL, MC-PAMEKASAN\nTT Number: INC-20250524-00000232\n*▪Observed :*\n▪ 5 SITES DOWN UNDER SITE 16PMN0015_TEJA_BARAT_PAMEKASAN_PL, MC-PAMEKASAN\nTT | WO: INC-20250524-00000232\nLevel : MAJOR\n \n*▪Start Time :*\n2025-05-24 00:18:40\n-\n\n*▪Service Impact :*\n▪Site Down:\n2G = 5 BTS\n4G = 5 eNodeB\n\n*▪MC Cluster Impacted :*\nMC-PAMEKASAN\n\n*▪Site list impact :*\nSite Name\n16PMN0015_TEJA_BARAT_PAMEKASAN_PL\n16PMN0013_PANGJAJAR_MT\n16PMN0011_BUDAKAN_PMK_TB\n16PMN0016_NYALABU_TB\n16PMN0009_PROPPO_EP',
            'ground_truth': '接入环双点开环造成上游传输路径中断，引发多站点断站：\n故障点1：\n根因描述：\nSite 16PMN00092025-05-23~2025-05-24期间，多次发生市电中断，引发站点业务中断。\n根因告警：\nMains Failure;PROPPO_EP_TR\n\n故障点2：\n根因描述：\n2025-05-24 00:16:13 Site 16PMN0026路由器MDR-PLGR-OPT-H910C和Site 16PMN0015路由器MDR-TEBP-OPT-H910C之间的物理连接中断。\n根因告警：\nPhysical Port Down;MDR-PLGR-OPT-H910C_GigabitEthernet0/2/0\nPhysical Port Down;MDR-PLGR-OPT-H910C_GigabitEthernet0/2/2\nPhysical Port Down;MDR-TEBP-OPT-H910C_GigabitEthernet0/2/1\nPhysical Port Down;MDR-TEBP-OPT-H910C_GigabitEthernet0/2/3',
        },
        {
            'data_source': 'ioh',
            'question': '故障工单信息如下：\nTT Title: *#6 CLOSE MAJOR [DIOC ALERT] 5 SITES DOWN UNDER SITE 16PMN0015_TEJA_BARAT_PAMEKASAN_PL, MC-PAMEKASAN, JAVA*\n\n*▪Observed :*\n▪ 5 SITES DOWN UNDER SITE 16PMN0015_TEJA_BARAT_PAMEKASAN_PL, MC-PAMEKASAN\nTT | WO: INC-20250524-00001457\nLevel : MAJOR\n \n*▪Start Time :*\n2025-05-24 01:53:40\n\n*▪Service Impact :*\n▪Site Down:\n2G = 5 BTS\n4G = 5 eNodeB\n\n*▪MC Cluster Impacted :*\nMC-PAMEKASAN\n\n*▪Site list impact : *\nSite Name\n16PMN0015_TEJA_BARAT_PAMEKASAN_PL\n16PMN0013_PANGJAJAR_MT\n16PMN0011_BUDAKAN_PMK_TB\n16PMN0016_NYALABU_TB\n16PMN0009_PROPPO_EP',
            'ground_truth': '接入环双点开环造成传输路径中断，引发多站点断站：\n故障点1：\n根因描述：\nSite 16PMN0009 2025-05-23~2025-05-24期间，多次发生市电中断，引发站点业务中断。\n根因告警：\nMains Failure;PROPPO_EP_TR\n\n故障点2：\n根因描述：\n2025-05-24 00:16:13 Site 16PMN0026路由器MDR-PLGR-OPT-H910C和Site 16PMN0015路由器MDR-TEBP-OPT-H910C之间的物理连接中断。\n根因告警：\nPhysical Port Down;MDR-PLGR-OPT-H910C_GigabitEthernet0/2/0\nPhysical Port Down;MDR-PLGR-OPT-H910C_GigabitEthernet0/2/2\nPhysical Port Down;MDR-TEBP-OPT-H910C_GigabitEthernet0/2/1\nPhysical Port Down;MDR-TEBP-OPT-H910C_GigabitEthernet0/2/3',
        },
        {
            'data_source': 'ioh',
            'question': '故障工单信息如下：\nTT Title: 6 SITE DOWN UNDER HUB SITE 16SPG0137_KLOBUR_PL MC-BANGKALAN\n\nTT : INC-20250524-00007826\nLevel : MAJOR\n\n*▪Start Time:*\n09:16:33 2025-05-24\n\n*▪Service Impact:*\nSite Down:\n2G = 6 BTS \n4G = 6 eNodeB \n\n*▪MC Cluster Impacted:*\nMC-BANGKALAN\n\n*▪Site list impact:*\nSite Name\n16SPG0003_SRESEH_EP\n16SPG0002_SRESEH_NOREH_TB\n16SPG0137_KLOBUR_PL\n16SPG0006_LABUHAN_SRESEH_TB\n16SPG0005_LABUHANSAMPANG_ST\n16BKN0145_PTI_PAENGBANGKALAN_PL',
            'ground_truth': '多站市电中断，引发路由器下电，批量站点退服\n故障点1：\n根因描述：\n2025-05-24 09:14:55 Site 16SPG0137路由器下电引发业务中断；\n根因告警：\nDevice Powered Off;MDR-KLOB-OPT-H910C\n\n故障点2：\n根因描述：\n2025-05-24 09:14:55 Site 16BKN0145路由器下电引发业务中断；\n根因告警：\nDevice Powered Off;MDR-PPAE-OPT-H910C\n\n故障点3：\n根因描述：\n2025-05-24 09:14:27 Site 16BKN0160路由器下电引发业务中断；\n根因告警：\nMains Failure;20BKN131_PENJALINAN_BKN',
        },
        {
            'data_source': 'ioh',
            'question': '故障工单信息如下：\nTT Title: 7 SITE DOWN UNDER HUB SITE 16SPG0137_KLOBUR_PL MC-BANGKALAN\n\nTT : INC-20250524-00020832\nWO : CM-20250524-00010090\nLevel : MAJOR\n\n*▪Start Time:*\n2025-05-24 15:24:30\n\n*▪Service Impact:*\nSite Down:\n2G = 7 BTS \n4G = 7 eNodeB \n\n*▪MC Cluster Impacted:*\nMC-BANGKALAN\n\n*▪Site list impact:*\n16SPG0003_SRESEH_EP\n16SPG0002_SRESEH_NOREH_TB\n16SPG0137_KLOBUR_PL\n16SPG0006_LABUHAN_SRESEH_TB\n16SPG0005_LABUHANSAMPANG_ST\n16BKN0145_PTI_PAENGBANGKALAN_PL\n16BKN0160_PENJALINAN_BKN_PS ',
            'ground_truth': '多站市电中断，引发路由器下电，批量站点退服\n故障点1：\n根因描述：\n2025-05-24 15:22:25 Site 16SPG0145路由器下电引发业务中断；\n根因告警：\nDevice Powered Off;MDR-PPAE-OPT-H910C\n\n故障点2：\n根因描述：\n2025-05-24 15:22:25 Site 16SPG0137路由器下电引发业务中断；\n根因告警：\nDevice Powered Off;MDR-KLOB-OPT-H910C\n\n故障点3：\n根因描述：\n2025-05-24 15:22:37 Site 16BKN0160路由器下电引发业务中断；\n根因告警：\nMains Failure;20BKN131_PENJALINAN_BKN',
        },
        {
            'data_source': 'ioh',
            'question': '故障工单信息如下：\nTT Title: 7 SITE DOWN UNDER HUB SITE 16BKN0190_TELLOK_BKN_PS, MC-BANGKALAN & MC-SAMPANG, JAVA\n\nTT : INC-20250524-00021573\nLevel : MAJOR\n\n*▪Start Time:*\n 2025-05-24 15:49:29\n\n\n*▪Service Impact:*\nSite Down:\n2G = 7 BTS \n4G = 7 eNodeB \n\n*▪MC Cluster Impacted:*\nMC-BANGKALAN & MC-SAMPANG\n\n*▪Site list impact:*\nSite Name\n16SPG0019_KEPAY_ST\n16SPG0012_GNSANDANGAN_TB\n16BKN0190_TELLOK_BKN_PS\n16BKN0141_TAMBAK_MDR_KN\n16BKN0193_KARANG_PANASAN_BLEGA_PL\n16BKN0197_LOMBANG_DAJAH_TB\n16BKN0155_JRENGIK_TB',
            'ground_truth': '多站市电中断，引发路由器下电，批量站点退服\n故障点1：\n根因描述：\n2025-05-24 13:00:12 Site 16SPG0019路由器下电引发业务中断；\n根因告警：\nDevice Powered Off;MDR-KPAY-OPT-H910D\n\n故障点2：\n根因描述：\n2025-05-24 15:47:28 Site 16BKN0190路由器下电引发业务中断；\n根因告警：\nMains Failure;20BKN122_TELLOK_BKN',
        },
    ] * 40

    start_time = time.time()
    results = asyncio.run(engine.execute_tasks(tasks[:]))
    logger.info(f"=======time: {time.time() - start_time}")

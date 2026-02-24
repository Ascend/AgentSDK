# 命令行接口说明<a name="ZH-CN_TOPIC_0000002469525620"></a>

## agentic\_rl<a name="ZH-CN_TOPIC_0000002469365676"></a>

**参数说明<a name="section973741317611"></a>**

|参数名|类型|说明|
|--|--|--|
|--config-path|--config-path|配置文件的路径。该配置文件的类型为yaml类型。|


**配置文件内容<a name="section61851418129"></a>**

| 参数名                       | 类型   | 说明                                                                 |
|---------------------------|------|--------------------------------------------------------------------|
| tokenizer_name_or_path    | str  | 分词器路径，需要包含完整的huggingface格式模型。                                      |
| train_iters               | int  | 训练迭代数。该值必须大于0。                                                     |
| agent_name                | str  | Agent的名称。必须以字母开头，由大小写字母，数字和下划线组成。                                  |
| agent_engine_wrapper_path | str  | 继承BaseEngineWrapper类的文件路径。                                         |
| dataset_additional_keys   | list | 设置数据集使用的字段(mindspeed-rl后端)                                         |
| mindspeed_rl              | dict | 参考下方mindspeed_rl参数                                                 |

mindspeed-rl后端参数

| 参数名 | 类型 | 说明 |
|---------------------------|------|--------------------------------------------------------------------|
| data_path | str | 数据集路径，json格式的数据集，通过dataset_additional_keys指定使用字段。 |
| test_data_path | str | 测试集路径，json格式的数据集，通过dataset_additional_keys指定使用字段，不使用则置空。 |
| load_params_path          | str  | 训练模型的文件路径，需要包含完整的Megatron格式模型。参考[快速入门](quick_start.md#快速入门)完成权重转换。 |
| save_params_path          | str  | 训练模型的文件保存路径。                                                       |

> [!NOTE] 说明
>本章节中涉及到的所有路径，需满足以下要求：
>-   该路径必须存在。
>-   该路径下的文件夹权限要求750，文件权限要求为640。
>-   该路径不能为软链接。
>-   该路径的字符串长度不能大于1024。
>-   该路径的属主为当前用户。



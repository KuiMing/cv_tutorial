# Computer Vision Line Chat Bot

## Resource

- Azure App Service: B1

## App service setting

- TLS/SSL setting: 
    - HTTPS only: on 

## Prepare 

1. Open SSH session in browser

2. edit `/home/config.json`
```
{
    "line":{
            "line_secret":...,
            "line_token":...
    },
    "azure":{
            "subscription_key":...,
            "endpoint":"https://<your name of Azure Cognitive Services>.cognitiveservices.azure.com/"
    },
    "imgur":{
            "client_id":...,
            "client_secret":...,
            "access_token":...,
            "refresh_token":...
    }
}
```
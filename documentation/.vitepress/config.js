export default {

    lang: 'en-US',

    title: 'Snow and Cloud Segmentation',
    description: 'Deep Learning for Accurate Snow and Cloud Segmentation in Alpine Landscapes',

    themeConfig: {

        nav: [
            {text: 'Start', link: '/index'},
            {text: 'Getting Started', link: '/docs/getting-started'}
        ],

        sidebar: {
            '/docs/': [

                {
                    text: 'Working Pipeline',
                    items: [
                        {text: 'Getting Started', link: 'docs/getting-started'},
                        {text: 'Pipeline', link: 'docs/pipeline'},
                    ],
                    collapsible: true,
                    collapsed: false,
                },

                {
                    text: 'Data Sources',
                    items: [
                        {text: 'ExoLabs', link: 'docs/ExoLabs_Classifications'},
                        {text: 'Sentinel2 (Bands)', link: 'docs/sentinel2_bands'},
                        {text: 'Landsat8', link: 'docs/landsat8'},
                    ],
                    collapsible: true,
                    collapsed: true,
                },

                {
                    text: 'Nice to Know',
                    items: [
                        {text: 'Python and Conda', link: 'docs/python_and_conda'},
                        {text: 'Working with Euler', link: 'docs/euler'},
                        {text: 'Working with Remote Desktop', link: 'docs/remote_desktop'},
                        {text: 'Improve VPN Stability', link: 'docs/improve-vpn'},
                    ],
                    collapsible: true,
                    collapsed: true,
                },

                {
                    text: 'Data Preparation',
                    items: [
                        {text: 'Data Source', link: 'docs/data_sources'},
                        {text: 'Augmentation and Sampling', link: 'docs/augmentation_and_sampling'},
                        {text: 'Hand Annotations', link: 'docs/hand_annotations'},
                        {text: 'Automated Annotations', link: 'docs/automated_annotations'},
                    ],
                    collapsible: true,
                    collapsed: true,
                },

                {
                    text: 'Models',
                    items: [
                        {text: 'Overview', link: 'docs/models/algorithms'},
                        {text: 's2cloudless', link: 'docs/models/s2cloudless'},
                        {text: 'Unet', link: 'docs/models/unet'},
                    ],
                    collapsible: true,
                    collapsed: true,
                },
            ]
        },


    }
}

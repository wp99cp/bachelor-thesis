export default {

    lang: 'en-US',
    lastUpdated: true,


    title: 'Snow & Cloud Segmentation',
    description: 'Deep Learning for Accurate Snow and Cloud Segmentation in Alpine Landscapes',

    themeConfig: {

        nav: [
            {text: 'Start', link: '/index'},
            {text: 'Getting Started', link: '/docs/working_pipeline/getting-started'}
        ],

        sidebar: {

            '/docs/': [

                {
                    text: '',
                    items: [
                        {text: 'Getting Started', link: '/docs/working_pipeline/getting-started'},
                        {text: 'Pipeline', link: '/docs/working_pipeline/pipeline'},
                    ],
                    collapsible: true,
                    collapsed: false,
                },


                {
                    text: 'Data Sources',
                    items: [
                        {text: 'Data Sources - Overview', link: '/docs/datasources/datasources'},
                        {text: 'Sentinel-2', link: '/docs/datasources/sentinel-2'},
                        {text: 'Landsat-8', link: '/docs/datasources/landsat-8'},
                        {text: 'ExoLabs', link: '/docs/datasources/exolabs'},
                    ],
                    collapsible: true,
                    collapsed: true,
                },


                {
                    text: 'Data Preparation',
                    items: [
                        {text: 'Data Preparation - Overview', link: '/docs/pre-processing/data_processing'},
                        {text: 'Hand Annotations', link: '/docs/pre-processing/hand_annotations'},
                        {text: 'Automated Annotations', link: '/docs/pre-processing/automated_annotations'},
                        {text: 'Clean Up Masks', link: '/docs/pre-processing/clean_up_masks'},

                        {text: 'Augmentation and Sampling', link: '/docs/pre-processing/augmentation_and_sampling'},
                    ],
                    collapsible: true,
                    collapsed: true,
                },

                {
                    text: 'Training and Models',
                    items: [
                        {text: 'Overview', link: '/docs/models/algorithms'},
                        {text: 's2cloudless', link: '/docs/models/s2cloudless'},
                        {text: 'Unet', link: '/docs/models/unet'},
                    ],
                    collapsible: true,
                    collapsed: true,
                },

                {
                    text: 'Nice to Know',
                    items: [
                        {text: 'Python and Conda', link: '/docs/nice_to_know/python_and_conda'},
                        {text: 'Working with Euler', link: '/docs/nice_to_know/euler'},
                        {text: 'Working with Remote Desktop', link: '/docs/nice_to_know/remote_desktop'},
                        {text: 'Improve VPN Stability', link: '/docs/nice_to_know/improve-vpn'},
                        {text: 'Remote Machine', link: '/docs/nice_to_know/remote_machine'},
                    ],
                    collapsible: true,
                    collapsed: true,
                },


            ]
        },

    }
}

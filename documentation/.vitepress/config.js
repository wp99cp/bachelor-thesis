export default {

    lang: 'en-US',

    title: 'Snow and Cloud Segmentation',
    description: 'Deep Learning for Accurate Snow and Cloud Segmentation in Alpine Landscapes',

    themeConfig: {

        nav: [
            {text: 'Start', link: '/index'},
            {text: 'Getting Started', link: '/docs/working_pipeline/getting-started'}
        ],

        sidebar: {
            '/docs/': [

                {
                    text: 'Working Pipeline',
                    items: [
                        {text: 'Getting Started', link: 'docs/working_pipeline/getting-started'},
                        {text: 'Pipeline', link: 'docs/working_pipeline/pipeline'},
                    ],
                    collapsible: true,
                    collapsed: false,
                },

                {
                    text: 'Data Sources',
                    items: [
                        {text: 'ExoLabs', link: 'docs/datasources/ExoLabs_Classifications'},
                        {text: 'Sentinel2 (Bands)', link: 'docs/datasources/sentinel2_bands'},
                        {text: 'Landsat8', link: 'docs/datasources/landsat8'},
                    ],
                    collapsible: true,
                    collapsed: true,
                },

                {
                    text: 'Nice to Know',
                    items: [
                        {text: 'Python and Conda', link: 'docs/nice_to_know/python_and_conda'},
                        {text: 'Working with Euler', link: 'docs/nice_to_know/euler'},
                        {text: 'Working with Remote Desktop', link: 'docs/nice_to_know/remote_desktop'},
                        {text: 'Improve VPN Stability', link: 'docs/nice_to_know/improve-vpn'},
                    ],
                    collapsible: true,
                    collapsed: true,
                },

                {
                    text: 'Data Preparation',
                    items: [
                        {text: 'Data Source', link: 'docs/pre-processing/data_sources'},
                        {text: 'Augmentation and Sampling', link: 'docs/pre-processing/augmentation_and_sampling'},
                        {text: 'Hand Annotations', link: 'docs/pre-processing/hand_annotations'},
                        {text: 'Automated Annotations', link: 'docs/pre-processing/automated_annotations'},
                        {text: 'Clean Up Masks', link: 'docs/pre-processing/clean_up_masks'},
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

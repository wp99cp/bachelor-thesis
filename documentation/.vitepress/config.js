export default {

    lang: 'en-US',

    title: 'Snow and Cloud Segmentation',
    description: 'Deep Learning for Accurate Snow and Cloud Segmentation in Alpine Landscapes',

    themeConfig: {

        sidebar: {
            '/docs/': [

                {
                    text: 'Data Sources and Pre-Processing',
                    items: [
                        {text: 'Getting Started', link: 'docs/getting-started'},
                        {text: 'ExoLabs', link: 'docs/ExoLabs_Classifications'},

                    ]
                },

                {
                    text: 'Nice to Know',
                    items: [
                        {text: 'Python and Conda', link: 'docs/python_and_conda'},
                        {text: 'Working with Euler', link: 'docs/euler'},
                        {text: 'Working with Remote Desktop', link: 'docs/remote_desktop'},
                        {text: 'Sentinel2 Bands', link: 'docs/sentinel2_bands'},
                    ]
                },

                {
                    text: 'Data Preparation',
                    items: [
                        {text: 'Data Source', link: 'docs/data_sources'},
                        {text: 'Augmentation and Sampling', link: 'docs/augmentation_and_sampling'},
                        {text: 'Hand Annotations', link: 'docs/hand_annotations'},

                    ]
                },

                {
                    text: 'Models',
                    items: [
                        {text: 'Overview', link: 'docs/models/algorithms'},
                        {text: 's2cloudless', link: 'docs/models/s2cloudless'},
                        {text: 'Unet', link: 'docs/models/unet'},
                    ]
                }


            ]
        },

        nav: [
            {text: 'Start', link: 'index'},
            {text: 'Getting Started', link: 'docs/getting-started'}
        ]
    }
}

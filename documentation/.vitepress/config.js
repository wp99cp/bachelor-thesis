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
                    ]
                },

                {
                    text: 'Nice to Know',
                    items: [
                        {text: 'Python and Conda', link: 'docs/python_and_conda'},
                    ]
                },


            ]
        },

        nav: [
            {text: 'Start', link: 'index'},
            {text: 'Getting Started', link: 'docs/getting-started'}
        ]
    }
}
